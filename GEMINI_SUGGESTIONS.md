# Report Analisi Codebase e Suggerimenti per ik_llama.cpp

In base all'analisi del progetto descritto nel `PROJECT_CONTEXT.md` (aggiunta del tier su NVMe per il prompt cache) e all'ispezione della codebase (con focus sui moduli server e cache), sono state rilevate le seguenti criticità, bug e aree di miglioramento.

## 1. Bug Critici nella Logica di Cache Attuale

*   **Move Distruttivi in Ricerca (`server-task.cpp`):**
    Nel metodo `server_prompt_cache::load()`, c'è un difetto in cui `it->tokens` viene spostato (`std::move`) in una variabile locale se `think_tokens.exclude` è false. Questo comportamento distrugge (svuota) i token di *ogni* entry della cache visitata durante la ricerca, anche se non viene poi selezionata come "migliore". Un problema simile esiste per la gestione della cache interna dello slot.
    *Suggerimento:* Utilizzare riferimenti (reference) o chiamare `.clone()` invece di fare un `move` degli oggetti in memoria se devono rimanere validi nella cache.

*   **Logica Errata sui "Thinking Tokens":**
    Sempre in `load()`, la ricerca estrae i `think_tokens` in base al *primo* elemento della cache incontrato per filtrare la query di ricerca. Questo porta a valutazioni errate nel caso in cui le varie entry nella cache abbiano configurazioni diverse, o se la cache è vuota all'inizio.

*   **Assegnazioni Duplicate (`server-task.h`):**
    Il metodo `server_prompt::from_json` contiene un'assegnazione duplicata per il campo `n_kept_prompt`. Si consiglia di pulire questo metodo per evitare errori quando verranno aggiunti i nuovi campi (come `disk_file`).

## 2. Colli di Bottiglia nelle Performance (RAM-tier attuale)

*   **Cicli inefficaci in Allocazione (`server-task.cpp`):**
    Nel metodo `server_prompt_cache::alloc()`, il sistema attualmente esegue una copia clonata dei prompt token e un match costoso del prefisso comune (Longest Common Prefix - LCP) per *ogni singola entry* presente nella cache. Essendo un'operazione O(N_cache * Length_prompt), causerà rallentamenti significativi nelle allocazioni degli slot qualora la cache dovesse crescere.

*   **Detokenizzazione Ridondante (`server-common.cpp`):**
    La logica attuale di matching del prefisso (LCP) esegue una detokenizzazione dalle sequenze di token alle stringhe di testo. Questa operazione all'interno dei cicli è molto costosa.
    *Suggerimento:* Ottimizzare cachando la rappresentazione in stringa o, meglio ancora, fare affidamento a un matching diretto nello spazio dei token in combinazione con l'allineamento dei blocchi (boundary trim & align), come già parzialmente discusso nella fase "Idee da rubare a ds4" del piano.

## 3. Lacune Architetturali nel Piano per il "Disk Tier"

Il piano delineato in `PROJECT_CONTEXT.md` tralascia alcune casistiche importanti per l'ambiente di produzione:

*   **Blocco Sincrono (I/O Blocking) del Thread Principale:**
    Implementare `dump_to_disk` e `restore_from_disk` in modo sincrono direttamente in `update()` o `load()` (eseguiti durante il ciclo di inferenza principale) causerà blocchi considerevoli. Il trasferimento sincrono su disco NVMe, benché veloce, unito al parsing e all'attesa del filesystem (specie con mmap attivo per i pesi modello) bloccherà le inferenze in corso per *tutti* gli altri utenti concorrenti del server.
    *Suggerimento:* Gestire i dump (eviction su disco) e i restore tramite un thread di background o I/O asincrono per non generare spike di latenza.

*   **Gestione Incompleta dello Stato (Multimodalità / Checkpoints):**
    Il piano si concentra sui file KV puri. Tuttavia la struct `server_prompt` contiene anche `checkpoints` (liste utilizzate per i recurrent models o contesti complessi). Inoltre, l'oggetto `server_tokens` può racchiudere dati derivati da media/immagini (vision). Una demotion elementare del file KV provocherebbe la perdita della cache non-testuale, invalidando il contesto.
    *Suggerimento:* Il "sidecar JSON" menzionato nello Step 4 del piano diventerà fondamentale; dovrà tracciare anche eventuali metadata visivi e serializeizzare i checkpoints prima dello spostamento su disco.

*   **Sovrascritture in fase di Persistenza (Bootstrapping):**
    Senza un'implementazione attenta, un contatore per l'ID dei file su disco (`next_disk_id` o `timestamp-counter`) ripartirà da zero a ogni avvio del server. Se il bootstrap (Step 6) non è gestito attentamente scannerizzando i file pregressi per riprendere il conteggio in modo sicuro, si rischierà di generare collisioni di nomi o di troncare stati precedenti.

*   **Overhead del LRU Scan:**
    Usare un semplice scan basato su `mtime` sui file di disco al momento dell'eviction di un nuovo elemento (in `update()`) causerà troppe letture I/O sul filesystem se la directory contiene migliaia di entry.
    *Suggerimento:* Mantenere in RAM un tracker leggero (es. priority queue o `std::list` aggiuntiva) per gli id attivi nel Disk Tier, o un indice in memoria per decidere la "vittima" senza toccare fisicamente i metadati della directory finché non serve fare unlink.