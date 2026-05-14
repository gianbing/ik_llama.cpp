# Disk-tier prompt cache

The server (`llama-server`) supports an optional NVMe-backed second tier for
the prompt cache. KV states evicted from RAM are written to disk instead of
being discarded, and restored on the next request that re-uses the same
prefix. This avoids cold-prefilling the whole prompt on session resume.

The implementation lives in:

- `examples/server/server-task.{h,cpp}` &mdash; `server_prompt_cache::{alloc,update,load,dump_to_disk,restore_from_disk,enforce_disk_limit,bootstrap_disk_tier}`
- `examples/server/server-context.{h,cpp}` &mdash; `server_slot::prompt_save` (boundary trim before extraction)
- `common/common.{h,cpp}` &mdash; CLI flag plumbing

## CLI flags

| Flag                       | Default        | Purpose                                            |
|----------------------------|----------------|----------------------------------------------------|
| `--cache-disk PATH`        | empty (off)    | Directory where `.kv` files are written            |
| `--cache-disk-mib N`       | `0` (no limit) | LRU upper bound on total disk usage, in MiB        |
| `--cache-disk-n-min N`     | `4096`         | Skip disk demotion below this many tokens          |
| `--cache-ram N`            | existing       | RAM-tier limit in MiB; set low to force demotion   |
| `-ctk`, `-ctv`             | existing       | KV quantization; `q4_0` gives compact `.kv` files  |
| `--flash-attn on`          | existing       | Required on most non-MLA models for sane KV size   |

A prompt is written to disk when it gets evicted from RAM &mdash; i.e. the RAM
tier is full and the prompt is the oldest non-disk-resident entry. On the
next request the slot is selected by prefix similarity (`calculate_slot_similarity`),
the state is loaded back from the file via `llama_state_seq_set_data`, and
the diverging tail (if any) is prefilled normally.

## Example

```bash
./build/bin/llama-server \
    -m model.gguf -ngl 99 \
    --cache-ram 64 \
    --cache-disk /tmp/kv-cache \
    --cache-disk-mib 20480 \
    --cache-disk-n-min 100 \
    -ctk q4_0 -ctv q4_0 --flash-attn on
```

`--cache-ram 64` keeps the RAM tier deliberately small so eviction to disk
happens after a couple of prompts; production deployments will normally
leave RAM higher and let only long-lived/cold entries demote.

## `.kv` file format (v2)

40-byte header followed by token IDs, then a single KV blob, then optionally
a sequence of checkpoint records.

| offset | size | field                  | notes                                                |
|-------:|-----:|------------------------|------------------------------------------------------|
|   0    |   4  | `magic = "IKKV"`       |                                                      |
|   4    |   1  | `version = 2`          |                                                      |
|   5    |   1  | `save_reason`          | `0=unknown, 1=cold, 2=continued, 3=evict, 4=shutdown`|
|   6    |   1  | `checkpoint_count`     |                                                      |
|   7    |   1  | reserved               |                                                      |
|   8    |   4  | `token_count`          | aligned by boundary trim, see below                  |
|  12    |   4  | `n_kept_prompt`        |                                                      |
|  16    |   4  | `n_discarded_prompt`   |                                                      |
|  20    |   8  | `data_size`            | size of the KV blob in bytes                         |
|  28    |   8  | `creation_time_us`     |                                                      |
|  36    |   4  | reserved               |                                                      |

Then `token_count * sizeof(llama_token)` bytes of token IDs, `data_size`
bytes of KV blob, and `checkpoint_count` records of 32 bytes each followed
by their per-checkpoint data blobs.

Files with `version=1` (no checkpoints, no save_reason) are still loadable
for backward compatibility.

## Boundary alignment

Before saving, the live KV is truncated down to a 2048-token boundary
(`N_aligned = N & ~2047`) so prompts that differ only in the trailing tokens
collapse on the same cache entry. The disk entry then matches as a prefix of
any longer prompt sharing the first `N_aligned` tokens, and the partial-hit
restore path covers the divergent tail.

Guards:

- skipped entirely for multimodal prompts (`has_mtmd_data()`) &mdash; cutting
  inside an image chunk would orphan media refs;
- skipped if `N_aligned < max(2048, cache_disk_n_min)` &mdash; we never trim
  a prompt below the configured disk-tier threshold.

## Benchmarks

### Phase 1 &mdash; full match

DeepSeek V2 Lite, MLA, KV `q4_0`, flash-attn on (`benchmark_disk_cache.sh`):

| chunk tokens | cold (ms) | disk restore (ms) | speedup    |
|-------------:|----------:|------------------:|-----------:|
|         512  |    1311   |             892   |     1.5x   |
|        1024  |    1639   |              35   |    46.6x   |
|        2048  |    2103   |              42   |    50.5x   |
|        4096  |    5004   |              38   |  **132.7x**|

### Phase 1.5 &mdash; partial hit

Workload: one cold baseline ~4426 tokens, 6 fillers ~2224 tokens (forcing
RAM eviction), 5 variations sharing the 4096-aligned prefix with different
short tails. Model: DSCoder V2 Lite Q8_0, KV `q4_0`, flash-attn on, `--cache-ram 64`
(`bench_phase1_5.py`):

| step               | prompt_n | prompt_ms | speedup vs cold |
|--------------------|---------:|----------:|----------------:|
| P0 (cold)          |     4426 |    7424   |       1.0&times;|
| Pv1 (disk hit)     |      334 |     752   |       9.9&times;|
| Pv2..Pv5 (RAM hit) |        7 | 118&ndash;146 |      ~57&times;|

Per-restore overhead is dominated by I/O at this state size (~3 MiB/ms for
the MLA case, ~1 MiB/ms on a 35B non-MLA model). See the trim-fires log
output to validate that boundary alignment is engaging.

### Non-MLA: Phase 2 ROI check

Tested on Qwen 3.6 35B-A3B (non-MLA, GQA, V-cache transposed), KV `q4_0`,
flash-attn on, same workload: single observed disk restore took **76 ms /
85 MiB**, scaling with state size, not with a constant CPU offset.

The original Phase 2 design proposed a raw CUDA copy path to bypass
`llama_state_seq_set_data`'s de-shuffle, assuming the de-shuffle was the
bottleneck on non-MLA models. The measurement does not support that
assumption: even on non-MLA the restore stays under 100 ms, two orders of
magnitude below cold prefill, so a CUDA-copy kernel would save at most a
handful of milliseconds in absolute terms. Open case: KV F16 (~4&times;
state size) was not measured and might still warrant the optimization.

## Reproducing the benchmarks

- `benchmark_disk_cache.sh` &mdash; Phase 1 full-match measurement
- `bench_phase1_5.py` &mdash; Phase 1.5 partial-hit measurement (set `LLAMA_DISK_DIR` to override `/tmp/kv-cache-deepseek`)

Both expect a running `llama-server` on `127.0.0.1:8080` with `--cache-disk`
and a low `--cache-ram` to actually exercise the disk path.

## Operational notes

- The disk tier is **off by default**: omit `--cache-disk` and the server
  behaves exactly as before.
- `enforce_disk_limit` enforces `--cache-disk-mib` by deleting the oldest
  files when the directory exceeds the budget. There is no compaction;
  individual entries are dropped wholesale.
- `bootstrap_disk_tier` re-indexes the directory at startup so previously
  written entries survive across server restarts.
- The on-disk format is **not** stable across major server versions: bump
  `version` in the header and clear the directory if the layout changes.
