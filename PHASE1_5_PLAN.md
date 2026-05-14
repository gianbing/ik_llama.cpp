# Phase 1.5 — Antirez-style refinements

Branch: `feature/disk-prompt-cache`. Punto di partenza validato: tag `phase1-disk-tier-validated` (commit con le 3 fix che hanno sbloccato 46-132x speedup).

Scelta del 2026-05-14: invece di Phase 2 (raw CUDA copy, basso ROI per MLA+q4_0 come da Appendice B di `phase2_design.md`), implementiamo due piccole feature ispirate al repo `antirez/ds4` che alzano la cache hit rate senza toccare il path CUDA.

## Step A — Save-reason tagging

**Obiettivo:** popolare il byte `save_reason` nell'header `.kv` v2 (oggi sempre 0) e usarlo per scelte di restore/eviction più mirate.

**Valori:**
- `0 = unknown` (retrocompatibile)
- `1 = cold` — prima demotion dopo cache full
- `2 = continued` — re-save di un entry già esistente (rare, per ora)
- `3 = evict` — demoted perché RAM tier full (caso comune)
- `4 = shutdown` — flush at server exit

**File da toccare:**
- `examples/server/server-task.cpp` — `dump_to_disk()` riceve `save_reason` argomento, scrive nell'header. Caller (`update()`, eventuale shutdown hook) passa il motivo giusto.
- Opzionale: `restore_from_disk()` log del save_reason per debug.

**Acceptance:**
- Build pulita
- Test: avviare server, fare prompt, verificare con `xxd /tmp/kv-cache-deepseek/*.kv | head -3` che il byte offset 5 sia non-zero
- Backward-compatible: file v2 vecchi con `save_reason=0` ancora caricabili

**Effort stimato:** 1h

## Step B — Boundary trim + chunk alignment

**Obiettivo:** prima di `dump_to_disk`, allineare la lunghezza salvata a un multiplo di 2048 token e tagliare gli ultimi 32 token. Questo aumenta il match-rate al restore perché prompt simili condividono boundary "puliti" più spesso.

**Logica (versione effettivamente implementata):**
```
N         = tokens.size()
N_aligned = N & ~(2048 - 1)              // floor-only allineamento a 2048
n_min     = max(2048, cache_disk_n_min)  // non scendere sotto la soglia disk
if (!tokens.has_mtmd_data() &&
    N_aligned >= n_min &&
    N_aligned < N) {
    truncate tokens, KV state, checkpoints a N_aligned
}
```

Differenze rispetto alla bozza originale:
- **No `-32` di tail trim:** la formula `(N-32) & ~2047` collassa N=2048..2079 a 0 e N=4096 a 2048 (perdita 2048 token su un input esattamente allineato). Allineamento floor-only senza `-32` evita la regressione e mantiene l'obiettivo (collidere prompt con coda variabile entro l'ultimo blocco).
- **Soglia `n_min = max(align, n_min_disk)`:** evita di trimmare prompt che, dopo il trim, finirebbero sotto `cache-disk-n-min` (default 4096) e verrebbero scartati dal `dump_to_disk` invece di essere persistiti.
- **Skip su multimodale:** se `server_cached_prompt.tokens.has_mtmd_data()` il trim viene saltato — la coda potrebbe cadere in mezzo a un chunk immagine.

**File da toccare:**
- `examples/server/server-task.cpp` — `dump_to_disk()` calcola `N_aligned`, tronca `tokens`, e nei checkpoint filtra quelli con `pos_max >= N_aligned`.
- `server-task.h` — opzionale: flag CLI `--cache-disk-align N` (default 2048) e `--cache-disk-trim N` (default 32). Per ora hardcoded.

**Acceptance:**
- Benchmark prima/dopo: per prompt con lunghezza 4096+32, prima del fix il restore matchava esattamente, dopo si attiva il path "ricostruisci ultimi 32"; per prompt con lunghezza 4100, prima miss, dopo hit (perde 4 tok ma riusa 4096).
- No regressioni su `benchmark_disk_cache.sh` (4096 deve restare ≥100x)
- Edge case: prompt <= 32 token → skip dump completo

**Effort stimato:** 3h (include test e benchmark di conferma)

## Rollback plan

Se uno dei due step rompe il disk tier:
```bash
git reset --hard phase1-disk-tier-validated
```
Il branch torna esattamente allo stato del benchmark 46-132x.

## Non in scope per Phase 1.5

- **Dir-steering** (Step 3 dell'analisi antirez) — feature autonoma, valutiamo dopo. Effort 1-2 giorni, non legato al disk tier.
- **CUDA kernel optimizations** — richiederebbero lettura di `ds4_cuda.cu` (463KB) e porting non banale.
- **Cross-quant KV reuse** — troppo specifico per il loro formato.
- **Phase 2 originale (raw CUDA copy)** — bocciato in Appendice B per MLA+q4_0.

## Quando riprendere

Dopo compact / nuova sessione, leggi questo file + `phase2_design.md` Appendice B + memorie `project-disk-tier-status` e `ik-llama-launch-flags-moe-q4`. Punto di ingresso = `examples/server/server-task.cpp::dump_to_disk()`.

---

## Stato esecuzione (2026-05-14)

### Step A — Save-reason tagging ✅ DONE

**Modifiche:**
- `examples/server/server-task.h`: aggiunta enum `server_prompt_cache::disk_save_reason` (unknown/cold/continued/evict/shutdown). Firma `dump_to_disk(p, reason=unknown)`.
- `examples/server/server-task.cpp`: `hdr.save_reason = (uint8_t) reason` invece di hardcoded 0. `restore_from_disk` decodifica e logga il reason come stringa. Call site in `update()` passa `disk_save_reason::evict`.

**Verifica:** 5 entry dumpate durante stress test mostrano tutte `save_reason=0x03 (evict)` via `xxd -s 5 -l 1`. Build pulita. Backward-compat: file v2 vecchi con byte=0 → log "unknown".

### Step B — Boundary trim + chunk alignment ✅ DONE

**Decisione architetturale:** la plan originale ipotizzava il trim *in `dump_to_disk`*, ma `p.data` a quel punto è un blob KV monolitico estratto via `llama_state_seq_get_data` — non c'è modo di troncarlo retroattivamente senza un ctx KV vivo. Il trim va fatto **a `prompt_save` time**, prima di estrarre lo stato, chiamando `llama_kv_cache_seq_rm(ctx, id, N_aligned, -1)` sul KV vivo.

**Formula scelta:** `N_aligned = N & ~(align - 1)` con `align=2048` (senza il -32 della plan, che produrrebbe N_aligned=2048 per N=4096 perdendo 2048 token).

**Modifiche:**
- `examples/server/server-context.h`: `prompt_save` non più `const`.
- `examples/server/server-context.cpp`, `server_slot::prompt_save`: prima dell'estrazione, se `n_aligned >= align && n_aligned < n_full`:
  - `llama_kv_cache_seq_rm(ctx, id, n_aligned, -1)`
  - `cache_tokens.resize(n_aligned)` + `server_cached_prompt.tokens.resize(n_aligned)` per sincronia bookkeeping
  - `server_cached_prompt.checkpoints.remove_if(c.pos_max >= n_aligned)`
  - log `boundary trim: N -> N_aligned tokens (align=2048)`

**Verifica live:**

| Input tok | Atteso  | Osservato | OK |
|-----------|---------|-----------|-----|
| 1340      | no trim | 1340      | ✓   |
| 1790      | no trim | 1790      | ✓   |
| 1912      | no trim | 1912      | ✓   |
| 3574      | → 2048  | 2048      | ✓   |
| 3829      | → 2048  | 2048      | ✓   |
| 4777      | → 4096  | 4096      | ✓   |
| 5663      | → 4096  | 4096      | ✓   |

Demotion dopo 2 entry da ~34 MiB (4096 tok) → `/tmp/kv-cache-deepseek/prompt-0000000000000005.kv` con `token_count=0x00001000 (4096)`, `save_reason=0x03`. Header allineato perfettamente, restore size-check passerà.

**Cosa NON è stato fatto rispetto al plan originale:**
- CLI flag `--cache-disk-align N` / `--cache-disk-trim N` (hardcoded `align=2048`; il trim "−32 tail" è stato omesso perché produceva regressioni sui prompt esattamente allineati).
- Benchmark prima/dopo (`benchmark_disk_cache.sh`) — il bench dimostrerebbe l'incremento di hit-rate su prompt di lunghezza simile ma non identica.

**Benchmark partial-hit (2026-05-14, post-review fixes — commit `25c221e`):**

Workload: P0 baseline (~4426 tok) + 6 filler diversi (~2224 tok ciascuno) per
forzare RAM-eviction, poi 5 variations con stesso prefisso e coda diversa.
Modello: DeepSeek Coder V2 Lite Q8_0, KV q4_0 + flash-attn, --cache-ram 64,
--cache-disk-mib 20480, --cache-disk-n-min 100.

| step          | prompt_n | prompt_ms | note                              |
|---------------|---------:|----------:|-----------------------------------|
| P0 cold       |     4426 |    7424.1 | slot vuoto, disk vuoto            |
| filler avg    |    ~2200 |     ~3590 | cold (prompt nuovo ogni volta)    |
| Pv1 (1° hot)  |      334 |     752.6 | partial-hit disk: 4092 tok da .kv |
| Pv2 (RAM hit) |        7 |     126.1 | slot ha P0_trimmed in RAM         |
| Pv3 (RAM hit) |        7 |     122.1 |                                   |
| Pv4 (RAM hit) |        7 |     118.6 |                                   |
| Pv5 (RAM hit) |        7 |     145.8 |                                   |

Server log conferma 7 trim su save (P0 4426→4096; ogni filler 2224→2048) e
`disk restore: ... 4096 tokens, save_reason=evict, 11.07 ms` su Pv1.
Speedup vs cold: 9.9× (disk partial-hit) → 57× (RAM partial-hit).

Lo speedup grezzo (132× del Phase 1) era misurato su match esatto di tutto il
prompt; qui ogni variation ha coda diversa quindi paga il prefill della parte
non coperta. È esattamente il caso d'uso che Step B doveva sbloccare.

Script: `bench_phase1_5.py`.

### Commit suggerito

Non ancora committato. File modificati:
- `examples/server/server-task.h`
- `examples/server/server-task.cpp`
- `examples/server/server-context.h`
- `examples/server/server-context.cpp`

Rollback: `git reset --hard phase1-disk-tier-validated`
