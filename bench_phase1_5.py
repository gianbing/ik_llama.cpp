#!/usr/bin/env python3
"""bench_phase1_5.py — verifica che il trim Step B + i fix review dia partial-hit
su prompt che condividono il prefisso 4096-allineato e differiscono solo in coda.

Sequenza:
  1) baseline P0 ~4128 tok  → cold prefill (slot vuoto, disk vuoto)
  2) 6 filler ~2048 tok ognuno diversi → riempie RAM (limit 64 MiB), evicta P0 a disco
  3) 5 variations Pv1..Pv5 = P0[0:~4096 tok] + 32 token di "rumore" diverso ognuno
                          → ciascuno deve trovare un partial-hit ai 4096 tok salvati

Output:
  - tabella prompt_ms per ogni request
  - log conta delle entry su /tmp/kv-cache-deepseek/ (ci aspettiamo poche)
"""
import json
import os
import sys
import time
import urllib.request

SERVER = os.environ.get("LLAMA_SERVER", "http://127.0.0.1:8080")
DISK_DIR = "/tmp/kv-cache-deepseek"
N_FILLERS = 6
N_VARS = 5

# Una stringa di lorem prevedibile di ~5 char/token in inglese.
LOREM = (
    "The quick brown fox jumps over the lazy dog while the sleepy cat watches "
    "from a sunlit windowsill on a quiet afternoon in early autumn when leaves "
    "drift slowly downward to the cobblestone path below. "
) * 200

def make_prompt(seed_tag: str, tail_chars: str = "", base_chars: int = 22000) -> str:
    """Costruisce un prompt di ~4128 token: base condivisa + tag testa + coda variabile."""
    base = LOREM[:base_chars]
    return f"[seed:{seed_tag}] " + base + " " + tail_chars

def chat(content: str) -> dict:
    payload = json.dumps({
        "model": "local",
        "max_tokens": 1,
        "stream": False,
        "messages": [
            {"role": "system", "content": "Reply with one word only."},
            {"role": "user",   "content": content},
        ],
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{SERVER}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read())
    wall_ms = (time.perf_counter() - t0) * 1000.0
    timings = body.get("timings", {})
    return {
        "wall_ms":       wall_ms,
        "prompt_ms":     timings.get("prompt_ms", -1),
        "prompt_n":      timings.get("prompt_n", -1),
        "predicted_ms":  timings.get("predicted_ms", -1),
    }

def list_disk_entries():
    try:
        return sorted(os.listdir(DISK_DIR))
    except FileNotFoundError:
        return []

def fmt(label: str, r: dict) -> str:
    return (f"{label:<14} prompt_n={r['prompt_n']:>5}  "
            f"prompt_ms={r['prompt_ms']:>8.1f}  wall_ms={r['wall_ms']:>8.1f}")

def main():
    print(f"# server   : {SERVER}")
    print(f"# disk dir : {DISK_DIR}  (entries iniziali: {len(list_disk_entries())})")
    print()

    print("== 1) cold baseline P0")
    r0 = chat(make_prompt("BASE", tail_chars="end-marker-base"))
    print(fmt("P0 (cold)", r0))
    print()

    print(f"== 2) {N_FILLERS} filler ~2048 tok per forzare RAM eviction")
    short_base = LOREM[:11000]
    for i in range(N_FILLERS):
        rf = chat(f"[fill:{i}] {short_base} unique-{i}")
        print(fmt(f"F{i}", rf))
    print()
    print(f"   disk dir entries dopo filler: {len(list_disk_entries())}")
    for e in list_disk_entries():
        sz = os.path.getsize(os.path.join(DISK_DIR, e))
        print(f"     {e}  ({sz/1024/1024:.2f} MiB)")
    print()

    print(f"== 3) {N_VARS} variations di P0 (stesso prefisso, coda diversa)")
    for i in range(N_VARS):
        rv = chat(make_prompt("BASE", tail_chars=f"unique-suffix-num-{i}-end"))
        print(fmt(f"Pv{i+1} (hot)", rv))
    print()

    print(f"== finale: disk dir entries: {len(list_disk_entries())}")
    for e in list_disk_entries():
        sz = os.path.getsize(os.path.join(DISK_DIR, e))
        print(f"     {e}  ({sz/1024/1024:.2f} MiB)")

if __name__ == "__main__":
    sys.exit(main())
