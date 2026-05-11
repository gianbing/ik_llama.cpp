# Progetto: Disk Tier per il prompt cache di ik_llama.cpp (CUDA)

## Obiettivo
Aggiungere un **secondo tier su NVMe** al `server_prompt_cache` di
`ik_llama.cpp`, in modo che gli stati KV evictati dalla RAM non vengano
persi ma scritti su disco e ricaricabili nelle sessioni successive
senza ri-prefill. Target primario: DeepSeek-V4 Flash q2 su CUDA con
mmap dei pesi.

> **Revisione importante (2026-05-11):** il piano originale era "portare
> antirez/ds4 su ik_llama.cpp". Dopo aver letto il codice attuale si è
> visto che **non serve un porting**: ik_llama.cpp ha già tutta
> l'infrastruttura interna (prefix matching, eviction hook, save/restore
> API MLA-aware). Manca solo il backing su disco. Il delta è ~150 righe
> in 4 file, non un'implementazione da zero.

---

## Hardware target (invariato)

- CPU: i9-13900K (Raptor Lake, 8P+16E, no AVX-512)
- RAM: 64GB DDR5 6400
- GPU: RTX 3080 10GB (Ampere SM_86 → FlashMLA-3 compatibile)
- Storage: NVMe PCIe 5.0
- OS: CachyOS, KDE 6.x, fish shell
- **Memoria totale: ~74GB** (10GB VRAM + 64GB RAM)

## Vincoli di memoria (invariati)

| Componente | Dimensione |
|---|---|
| Pesi DS4 Flash q2 | ~81GB |
| OS + processi | ~4-6GB |
| KV cache 100k token | ~3-5GB |
| **Totale necessario** | **~88-92GB** |
| **Totale disponibile** | **~74GB** |

Deficit ~15-18GB → si usa **mmap** sui pesi (NVMe PCIe 5.0). MLA
mantiene il KV piccolo (~26GB a 1M token, pochi GB a 100k) → la latenza
NVMe del disk tier è tollerabile perché si trasferisce poco dato.

---

## Cosa c'è già in ik_llama.cpp (verificato sul codice)

| Feature | Stato | Riferimento |
|---|---|---|
| MLA per DeepSeek | ✅ | `src/graphs/build_deepseek2.cpp`, PR #188 |
| Flash Attention DeepSeek CUDA | ✅ | `ggml/src/ggml-cuda/fattn-*.cu`, PR #200 |
| FlashMLA-3 CUDA (Ampere+) | ✅ | `ggml/src/ggml-cuda/fattn-new-mma.cu`, PR #386 |
| Q8_0 KV cache + FlashMLA | ✅ | PR #265 |
| MoE ottimizzato CUDA | ✅ | `ggml/src/ggml-cuda/mmq_id.cu`, PR #283 |
| API `llama_state_seq_*` (MLA-aware) | ✅ | `include/llama.h:969-1001`, PR #497 |
| `llama_state_seq_save_file` / `_load_file` | ✅ | `include/llama.h:993,1000` |
| `server_prompt_cache` RAM-only con LCP matching | ✅ | `server-task.h:428` |
| Eviction hook + save automatico | ✅ | `server-context.cpp:1097-1120` |
| HTTP endpoints save/restore manuale per slot | ✅ | `server.cpp:884-980` |
| **Disk tier (LRU su NVMe, promotion/demotion)** | ❌ | **da implementare** |

HEAD attuale del repo: `3557b44` (#1777 "Avoid recurrent state copy").

---

## Architettura corrente del prompt cache (RAM)

**Strutture** (`examples/server/server-task.h:384-452`):
```cpp
struct server_prompt {
    server_tokens tokens;           // ID token (chiave per LCP matching)
    int n_kept_prompt;
    int n_discarded_prompt;
    thinking_tokens think_tokens;
    std::vector<uint8_t> data;      // blob KV serializzato (llama_state_seq_get_data)
    std::list<server_prompt_checkpoint> checkpoints;
};

struct server_prompt_cache {
    std::list<server_prompt> states;
    size_t limit_size;              // byte; 0 = no limit
    size_t limit_tokens;            // 0 = no limit
    llama_context* ctx;
    // load() / alloc() / update()
};
```

**Save trigger** (`server-context.cpp:1097-1120`):
- Dentro `get_available_slot`: se il candidato slot ha `f_keep <
  cache_ram_similarity` (sta per perdere contesto significativo) →
  `slot.prompt_save(*prompt_cache)`
- `prompt_save` (server-context.cpp:560):
  1. `llama_state_seq_get_size(ctx, id, 0)` → dimensione
  2. `prompt_cache.alloc(...)` → slot in `states`, evicta entry contenute
  3. `llama_state_seq_get_data(ctx, buf, size, id, 0)` → riempie `data`

**Load** (`server-task.cpp:1075-1135`):
- Itera `states`, calcola LCP + similarity con i nuovi token
- Sceglie il migliore, `llama_state_seq_set_data(ctx, ...)` per restore
- L'entry viene rimossa dalla lista (verrà ri-salvata al prossimo evict)

**LRU** (`server-task.cpp:1192-1219`):
- `pop_front()` finché `size() > limit_size`
- Tiene sempre almeno 1

---

## Piano implementativo

### Step 1 — Estensione dati (server-task.h)

```cpp
struct server_prompt {
    // ... campi esistenti ...
    std::string disk_file;          // != "" se data è su disco non in RAM
    uint64_t    data_disk_size = 0; // dimensione su disco (per limits)
};

struct server_prompt_cache {
    // ... campi esistenti ...
    std::string disk_dir;           // "" = disk tier disabilitato
    size_t      limit_disk_bytes = 0;
    // tutte le altre methods invariate, ne aggiungiamo due private:
    //   void dump_to_disk(server_prompt& p);
    //   bool restore_from_disk(server_prompt& p);
};
```

### Step 2 — Demotion in `update()` (server-task.cpp:1192)

```cpp
void server_prompt_cache::update() {
    // 1. RAM tier: invece di pop_front, dump_to_disk se disk_dir != ""
    if (limit_size > 0) {
        while (states.size() > 1 && size() > limit_size) {
            auto& victim = states.front();
            if (!disk_dir.empty() && victim.disk_file.empty()) {
                dump_to_disk(victim);   // scrive data[] su file, svuota victim.data
                // entry rimane in states, ma con data vuoto + disk_file populato
                states.splice(states.end(), states, states.begin()); // sposta in coda
            } else {
                states.pop_front();      // niente disk tier → vecchio comportamento
            }
        }
    }
    // 2. Disk tier: LRU su mtime dei file .kv quando supera limit_disk_bytes
    //    Scan disk_dir, ordina per mtime, unlink i più vecchi finché rientri
}
```

### Step 3 — Promotion in `load()` (server-task.cpp:1075)

```cpp
// dentro l'iterazione che cerca it_best:
// se it->disk_file != "" e data.empty(), restore_from_disk(*it) prima di set_data
if (it_best != states.end()) {
    if (!it_best->disk_file.empty() && it_best->data.empty()) {
        if (!restore_from_disk(*it_best)) {
            return false;
        }
    }
    // ... resto invariato (set_data, erase) ...
}
```

### Step 4 — Helper di serializzazione

```cpp
// formato file: header magic + sidecar JSON metadata + blob KV
// magic: "IKKV\x01"
// header (32 byte): magic[5] + reserved[3] + token_count(u32) +
//                   data_size(u64) + n_kept_prompt(u32) +
//                   n_discarded_prompt(u32) + creation_time(u64)
// payload: tokens[token_count]*u32 + data[data_size]
```

Naming file: `prompt-{timestamp}-{counter}.kv` (no SHA1: l'index dei
token vive in RAM tramite `server_prompt::tokens`, niente bisogno di
chiave content-addressed).

### Step 5 — CLI flags (common/common.h, common/common.cpp)

Modello: identico a `--cache-ram`.

```
--cache-disk PATH        directory per il tier disco ("" = disabled)
--cache-disk-mib N       limite spazio disco in MiB (default: 16384)
--cache-disk-n-min N     min token per scrivere su disco (default: 4096)
```

Posizioni:
- `common/common.h:474-476` (campi nello struct params)
- `common/common.cpp:2275` (arg parser)
- `common/common.cpp:2479` (help output)
- `examples/server/server-context.cpp:502-511` (passa al ctor)

### Step 6 — Bootstrap

All'avvio del server, se `disk_dir != ""`:
- Scansiona `disk_dir/*.kv`, parsa header, popola `states` con entry
  in modalità "disk-only" (`data` vuoto, `disk_file` settato, `tokens`
  letti dall'header)
- Cosa NON leggere: il blob KV (centinaia di MB) — solo i tokens
- Così il prefix matching del primo request già funziona contro la
  cache su disco

---

## Strategia di sviluppo

### Modello di sviluppo
**Non sviluppare direttamente su DS4 Flash q2.** Caricamento via mmap +
debug = troppo lento.

Usare un MoE/MLA piccolo che entra comodo in memoria:
- `DeepSeek-V2-Lite` (~16GB q4) — stessa famiglia MLA, ideale
- `Qwen3-30B-A3B` (~20GB q4) — MoE per stress-testare

Verificare disponibilità GGUF su HuggingFace prima di scaricare.

### Workflow consigliato
1. **Branch dedicato:** `feature/disk-prompt-cache` da `main`
2. **Step 1+4 prima:** definisci file format e helpers, testa con
   piccolo programma standalone (write/read di un blob fake)
3. **Step 2 (demotion):** unit test → fai partire il server con
   `--cache-ram 512 --cache-disk /tmp/kv`, vedi i file comparire
4. **Step 3 (promotion):** verifica che dopo restart del server con
   stesso prompt salta il prefill (`prompt cache load took ... ms` nel
   log invece di `prompt eval time`)
5. **Step 5+6:** boot dello stato della cache da file esistenti
6. **Stress test:** sessioni multiple parallele, eviction concorrente
7. **Solo dopo che funziona:** test su DS4 Flash q2 con mmap

### Test funzionali da superare
- [ ] Save→restart→resume su modello piccolo: nessun re-prefill se
      stessi token, prompt cache load < 1 sec
- [ ] Eviction RAM → file appare in `disk_dir`, file size coerente
- [ ] Eviction disco LRU: il più vecchio sparisce quando si supera
      `limit_disk_bytes`
- [ ] Crash recovery: se il server muore durante un dump, al restart
      gli `.kv` con header invalido vengono ignorati e rimossi
- [ ] Race: due slot che evictano contemporaneamente non si pestano i
      piedi (mutex su `disk_dir`? oppure naming univoco con counter
      atomico)

---

## Punti di attenzione tecnici

1. **`llama_state_seq_flags`**: solo 1 flag definito (`PARTIAL_ONLY=1`,
   `include/llama.h:786`). Il server passa `0` → stato completo,
   MLA-aware automatico. Non toccare.

2. **Dimensioni reali del blob**: il KV serializzato per MLA è molto
   compatto. Stima ordine di grandezza: ~16 KB per token su un modello
   DS4-Lite. A 100k token: ~1.6GB. Su NVMe PCIe 5.0 (~14 GB/s) la
   serializzazione/load è ~100ms — irrilevante rispetto al prefill
   evitato (decine di secondi).

3. **Tokens nell'header**: salvi token IDs come `u32` little-endian.
   A 100k token sono 400 KB — non significativi rispetto al blob.

4. **Atomicità di scrittura**: scrivi su `<file>.tmp` poi `rename()`.
   Su CachyOS (btrfs in genere) il rename è atomico. Niente
   `fsync()` esplicito per ora — accettiamo il rischio di perdita su
   power-loss.

5. **Eviction LRU sul disco**: scan della dir + sort per `mtime`. Non
   serve un index file: l'header del singolo `.kv` basta a
   ricostruire lo stato in RAM al boot.

6. **Cosa NON fare**: niente content-addressing SHA1 stile antirez.
   Il prefix matching avviene sui tokens già caricati in RAM. Il nome
   del file è solo uno UUID interno. Più semplice, più robusto a
   cambi di tokenizer.

7. **mmap dei pesi e disk KV concorrenti**: il process legge dai pesi
   mmappati e scrive nuovi file `.kv` sullo stesso NVMe. Verifica con
   `iostat` che non causi contention; eventualmente directory separate
   o nice/ionice sul thread di dump.

8. **I/O sync è una scelta deliberata, non un oversight** (vedi
   `GEMINI_SUGGESTIONS.md` §3a). Stima latenza per dev locale: KV ~200 MB
   su NVMe 5.0 (~14 GB/s) → ~14 ms di stallo del thread server. Per un
   server single-user con prefill da decine di ms o generazione, è
   trascurabile. Sotto carico concorrente multi-slot lo spike è reale →
   in quel caso si passa a un thread di background con una queue
   FIFO di dump pending (non urgente per Phase 1.5).

9. **Guard checkpoints/multimodal**: `dump_to_disk` rifiuta entry con
   `checkpoints` non vuoti (modelli recurrent: Mamba/RWKV/GLM-MTP) o
   `tokens.has_mtmd_data()` (vision/audio). Demoter solo `data[]`
   perderebbe stato e corromperebbe il restore. Per DS4 Flash text-only
   nessuna delle due condizioni si attiva. Estensione full → Phase 2.

10. **`next_disk_id` al bootstrap**: scan di `disk_dir/prompt-*.kv`,
    parse del counter dal nome, `next_disk_id = max(visti) + 1`. Evita
    collisioni di nome al restart. La LRU a runtime gira sulla list
    `states` in RAM (entry disk-resident hanno `data` vuoto e
    `disk_file` settato) — niente scan filesystem nel percorso caldo,
    solo al bootstrap e quando `enforce_disk_limit()` deve davvero
    unlinkare.

---

## File chiave da toccare

| File | Modifica |
|---|---|
| `examples/server/server-task.h` | +5 campi (`disk_file`, `disk_dir`, ecc.) |
| `examples/server/server-task.cpp` | modifica `update()` e `load()`, aggiungi `dump_to_disk` / `restore_from_disk` |
| `examples/server/server-context.cpp` | passa nuovi parametri al ctor di `server_prompt_cache` |
| `common/common.h` | +3 campi nei params |
| `common/common.cpp` | +3 entry nel parser, +3 nel help |

Stima totale: **~150 righe di codice nuovo, ~20 righe modificate**.

---

## Ambiente di sviluppo

- **Path repo:** `/home/gianluca/Claude/ik_llama.cpp` (btrfs, dopo
  migrazione da NTFS)
- **Build:** `./configure.sh && ./build.sh` — già funzionante
- **Compilatore:** gcc-15 (richiesto da nvcc 13.2), CUDA host
  compiler configurato
- **Target arch:** solo SM_86 (RTX 3080), niente fat binary
- **Shell fish — EOF heredoc non funziona**: usare file temporanei
- **No pip/python di sistema** (CachyOS PEP 668)
- **Standard di qualità:** prima di usare comandi/flag di ik_llama.cpp
  verificare che esistano nella versione corrente (HEAD evolve)

---

## Repo di riferimento

- `https://github.com/ikawrakow/ik_llama.cpp` — base (questo repo)
- `https://github.com/antirez/ds4` — implementazione disk-KV originale
  (Metal/llama.cpp), buona ispirazione per il file format ma il design
  qui è diverso
- `https://huggingface.co/antirez/deepseek-v4-gguf` — pesi DS4 Flash q2

---

## Idee da rubare a ds4 — Phase 1.5 (parcheggiate)

Decisione 2026-05-11: il design base **non** porta ds4 alla lettera (la maggior
parte è già implementata in `server_prompt_cache`). Ma due idee di antirez sono
utili e le aggiungiamo dopo aver finito le 6 fasi del piano principale:

### A. Boundary trim & chunk alignment al cold-save
**Cosa fa**: prima di scrivere il KV, taglia gli ultimi N token (default 32) e
allinea la lunghezza salvata a un multiplo di chunk (default 2048).

**Perché serve**: quando un client (Claude Code, opencode, ecc.) rimanda
l'intera conversazione + 1 nuovo turno, la tokenizzazione BPE ai bordi può
differire leggermente. Salvando una versione "allineata" del prefix, il match
LCP della request successiva ha probabilità molto più alta di centrare la
cache.

**Dove agganciare**: in `server_prompt_cache::alloc()` (server-task.cpp:1137),
prima di copiare il blob, ritagliare il `cur_size` corrispondente al nuovo
numero di token allineato. Serve un'API tipo
`llama_state_seq_get_data_partial(ctx, buf, size, id, flags, n_tokens)` —
verificare se `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` già lo supporta.

**Parametri CLI** (da aggiungere quando si implementa):
```
--cache-boundary-trim N    (default 32)
--cache-boundary-align N   (default 2048)
```

### B. Save-reason tagging nel file header
**Cosa fa**: nel header del file `.kv` aggiungere un byte con la causa del
save (0=unknown, 1=cold, 2=continued, 3=evict, 4=shutdown), come in ds4.

**Perché serve**: telemetria/debugging gratis. Permette di capire dal solo
file system se la cache è popolata da prefill lunghi (cold) o da rotazioni
forzate (evict).

**Dove agganciare**: il trigger è già differenziato nel server (cold = primo
save dopo prefill in `update_slots`, evict = save in `get_available_slot`).
Basta aggiungere un parametro alla signature di `dump_to_disk` e propagarlo.

**Header esteso** (40 byte, naturalmente allineato):
```cpp
struct kv_file_header {
    char     magic[4];          // "IKKV"
    uint8_t  version;           // 1
    uint8_t  save_reason;       // 0..4
    uint8_t  reserved[2];
    uint32_t token_count;
    uint32_t n_kept_prompt;
    uint32_t n_discarded_prompt;
    uint64_t data_size;
    uint64_t creation_time_us;
};
// sizeof = 40 (uint64 alignment forza padding di 4 byte dopo
// n_discarded_prompt). Niente packed/__attribute__: lo lasciamo
// allineato per evitare costi di accesso disallineato.
```

### Cosa NON rubiamo (e perché)

- **SHA1 content-addressing del filename**: ds4 usa `SHA1(tokens).kv`. Lookup
  O(1) per nome ma fragile a cambi di tokenizer. Il nostro scan O(N) all'avvio
  è banale per N piccoli.
- **Compressed indexer ratio-4**: kernel CUDA custom, vale solo se il KV MLA
  standard non basta. Phase 2 opzionale, non in roadmap.

---

## Ordine consigliato delle prossime sessioni con Claude Code

1. **Build verification:** verifica che `./build.sh` produca i binari,
   smoke test con `llama-cli --version`
2. **Scarica modello di dev:** DeepSeek-V2-Lite Q4 GGUF
3. **Baseline:** parti il server con `--cache-ram` solo, misura il
   comportamento attuale (log `prompt cache save took`,
   `prompt cache load took`)
4. **Branch + step 1+4:** scrivi le strutture e gli helpers di
   serializzazione, testali standalone
5. **Step 2 (demotion):** integra il dump nel `update()`
6. **Step 3 (promotion):** integra il restore nel `load()`
7. **Step 5+6:** CLI flags + boot dei file esistenti
8. **Test su DS4 Flash q2** (solo dopo che 1-7 funzionano)
