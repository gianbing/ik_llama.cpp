#!/usr/bin/env bash
# benchmark_disk_cache.sh
# Misura cold-prefill vs disk-restore latency per il disk-tier prompt cache di ik_llama.cpp.
#
# Metodo di misura:
#   - usa `timings.prompt_ms` dalla risposta JSON (tempo esatto di prefill lato server)
#   - cold prefill   : primo invio del prompt, KV state calcolato da zero
#   - evict da RAM   : 6 prompt filler (2048 tok) saturano i 256 MiB di cache RAM,
#                      forzando il dump del target su disco
#   - disk restore   : stesso prompt → KV state caricato dal file .kv su NVMe
#
# Requisiti:
#   - llama-server attivo su 127.0.0.1:8080 con --cache-disk e --cache-ram impostati
#   - python3 in PATH
#
# Variabili override:
#   LLAMA_SERVER=http://host:port   (default: http://127.0.0.1:8080)
#   N_EVICT_FILLERS=6               (filler per forzare eviction, default 6)

set -euo pipefail
export LC_NUMERIC=C

SERVER="${LLAMA_SERVER:-http://127.0.0.1:8080}"
N_FILLERS="${N_EVICT_FILLERS:-6}"
BENCH_TMPDIR=$(mktemp -d "${TMPDIR:-/tmp}/bench.XXXXXX")
trap 'rm -rf "$BENCH_TMPDIR"' EXIT

# ---------------------------------------------------------------------------
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

check_server() {
    curl -sf "$SERVER/health" -o /dev/null 2>/dev/null \
        || die "server non raggiungibile a $SERVER"
}

# gen_payload TOKENS SEED  → payload JSON su stdout
gen_payload() {
    python3 - "$1" "$2" <<'PYEOF'
import json, sys

n_tokens = int(sys.argv[1])
seed     = sys.argv[2]

# ~5.5 char/token per testo inglese prosa; sovrasto del 20% per sicurezza
chars = int(n_tokens * 1.2 * 5)
base  = "the quick brown fox jumps over the lazy dog " * 5000
text  = f"[seed:{seed}] " + base[:chars]

print(json.dumps({
    "model":      "local",
    "max_tokens": 1,
    "stream":     False,
    "messages": [
        {"role": "system", "content": "Reply with one word only."},
        {"role": "user",   "content": text},
    ],
}))
PYEOF
}

# post_and_get_timing PAYLOAD  → stampa prompt_ms dalla risposta JSON
post_and_get_timing() {
    local out="$BENCH_TMPDIR/resp.$$.json"
    curl -sf \
        -X POST "$SERVER/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$1" \
        -o "$out" 2>/dev/null || die "request failed"
    python3 - "$out" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
t = d.get('timings', {})
print(t.get('prompt_ms', -1))
PYEOF
    rm -f "$out"
}

# ---------------------------------------------------------------------------
check_server

printf '\n  Disk-tier prompt cache benchmark  —  server: %s\n\n' "$SERVER"
printf '  %-8s  %-18s  %-18s  %-8s\n' \
    TOKENS  'cold_prefill_ms'  'disk_restore_ms'  speedup
printf '  %-8s  %-18s  %-18s  %-8s\n' \
    '--------'  '------------------'  '------------------'  '--------'

CSV="$(pwd)/benchmark_results.csv"
printf 'chunk_tokens,cold_ms,disk_ms,speedup\n' > "$CSV"

for CHUNK_SIZE in 512 1024 2048 4096; do

    RUN_SEED="${CHUNK_SIZE}_$(date +%s%N)"

    # ---- 1. cold prefill ------------------------------------------------
    PAYLOAD=$(gen_payload "$CHUNK_SIZE" "$RUN_SEED")
    printf '  %-8s  cold prefill in corso...\r' "$CHUNK_SIZE"
    T_COLD=$(post_and_get_timing "$PAYLOAD")

    # ---- 2. evict target dalla RAM --------------------------------------
    # 256 MiB RAM / ~48 MiB per stato 2048-token ≈ 5 slot;
    # N_FILLERS filler garantiscono che il target venga demoted su disco.
    printf '  %-8s  eviction da RAM (%s filler)...\r' "$CHUNK_SIZE" "$N_FILLERS"
    for i in $(seq 1 "$N_FILLERS"); do
        FILLER=$(gen_payload 2048 "fill_${CHUNK_SIZE}_${i}_$(date +%s%N)")
        post_and_get_timing "$FILLER" > /dev/null
    done

    # ---- 3. disk restore ------------------------------------------------
    printf '  %-8s  disk restore in corso...\r' "$CHUNK_SIZE"
    T_DISK=$(post_and_get_timing "$PAYLOAD")

    SPEEDUP=$(python3 -c "
c=float('$T_COLD'); d=float('$T_DISK')
print(f'{c/d:.1f}x' if d > 0.1 else '>999x (near-zero)')
")

    printf '  %-8s  %-18.1f  %-18.1f  %-8s\n' \
        "$CHUNK_SIZE" "$T_COLD" "$T_DISK" "$SPEEDUP"
    printf '%s,%s,%s,%s\n' \
        "$CHUNK_SIZE" "$T_COLD" "$T_DISK" "$SPEEDUP" >> "$CSV"

done

printf '\n  Risultati salvati in: %s\n\n' "$CSV"
