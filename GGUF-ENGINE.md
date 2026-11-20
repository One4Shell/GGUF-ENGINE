# GGUFEngine

Classe riutilizzabile (browser, ES module) per usare modelli linguistici
**interamente lato client**, con una API semplice: passi il modello e ottieni
`load`, `generate`, `stream`, `chat` e `send` con cronologia.

Due backend, scelti automaticamente:

- **WebGPU** — modelli in formato MLC tramite
  [`@mlc-ai/web-llm`](https://github.com/mlc-ai/web-llm): usa la GPU quando il
  browser la espone, con prestazioni molto più alte;
- **WASM/CPU** — qualsiasi file `.gguf` tramite
  [`@wllama/wllama`](https://github.com/ngxson/wllama), sempre disponibile.

Con `backend: 'auto'` (default) il motore prova la GPU e, se WebGPU o il modello
GPU non sono disponibili, ripiega sul backend WASM senza interrompere l'app.

## Installazione offline (opzionale, consigliata)

Tutti gli asset runtime possono essere scaricati **sul server**, così l'applicazione
non contatta CDN o Hugging Face. Se però `vendor/` o il modello locale non sono
presenti, il motore ripiega automaticamente sulle fonti web **senza errori**:

```bash
bun run deps          # Node:  node scripts/download-deps.js
bun run deps:sh       # Bash:  bash scripts/download-deps.sh   (senza Node)
bun run serve         # oppure: node server.js
```

Lo script (Node e Bash fanno esattamente la stessa cosa) salva:

- `vendor/wllama/esm/index.js` e `vendor/wllama/esm/wasm/wllama.wasm` (Wllama 3.x);
- `vendor/wllama-compat/wasm/wllama.js` + `wllama.wasm` — fallback per Safari/browser senza JSPI/MEMORY64;
- `vendor/web-llm/lib/index.js` — runtime GPU self-hosted, **scaricato di default**;
- `models/MNN-Pocket-0.3B-Instruct-Q4_1.gguf` (~183 MiB).

Nota: i **pesi** dei modelli GPU (MLC) non sono inclusi nello script: WebLLM li
scarica la prima volta da Hugging Face e li conserva nella cache del browser.

Opzioni utili: `--skip-model`, `--skip-wllama`, `--skip-webllm`, `--skip-runtime`,
`--force`, `--check` (solo stato, non scarica), `--model-url=…`, `--model-file=…`,
`--wllama-version=…`, `--webllm-version=…`. Il download del modello è riprendibile.

Verifica rapida:

```bash
node scripts/download-deps.js --check     # oppure: bash scripts/download-deps.sh --check
```

Se il runtime GPU manca davvero, `--force` lo riscarica e il comando esce con
codice ≠ 0 quando una sezione fallisce (i download già completati restano).
`--with-webllm` è ancora accettato ma non serve più: WebLLM è il default —
`bun run deps:gpu` resta come alias di compatibilità.

`server.js` è un server statico minimale (nessuna dipendenza) che invia
`Cross-Origin-Opener-Policy: same-origin` e
`Cross-Origin-Embedder-Policy: require-corp`, necessari a WebGPU e ai thread
WASM, e supporta le richieste `Range` per riprendere i download.

```js
import { GGUFEngine } from './gguf-engine.js';

const llm = new GGUFEngine({
  modelUrl: './models/MNN-Pocket-0.3B-Instruct-Q4_1.gguf',
  systemPrompt: 'Sei un assistente utile, conciso e preciso.',
});

await llm.load();                                  // download + cache del browser
console.log(await llm.generate('Spiega WebAssembly in 2 frasi.'));
```

Demo pronta: apri `http://localhost:8080/` dopo `bun run serve`.

---

## API

### `new GGUFEngine(config)`

| Opzione | Default | Descrizione |
| --- | --- | --- |
| `backend` | `'auto'` | `'auto'`, `'webgpu'` o `'wasm'` (fissato al load) |
| `modelUrl` | — | URL del file `.gguf` (backend WASM; opzionale con `webgpuModel`) |
| `webgpuModel` | — | Id del modello MLC, es. `LFM2.5-350M-q4f32_1-MLC` |
| `webgpuModelRecord` | — | Record `ModelRecord` WebLLM completo (alternativa a `webgpuModel`) |
| `webgpuContextSize` | `4096` | Contesto dichiarato al backend GPU (token) |
| `webgpuFallbackToWasm` | `true` | Se WebGPU non è disponibile, usa il GGUF |
| `webllmModuleUrl` | vendor o CDN | Modulo ESM di WebLLM |
| `wllamaModuleUrl` | `./vendor/wllama/esm/index.js` | Modulo ESM di Wllama (locale) |
| `wasmPaths` | `{ default: './vendor/wllama/esm/wasm/wllama.wasm' }` | Mappa dei binari WASM (v3); accettate anche le chiavi legacy |
| `wllamaCompatPaths` | `./vendor/wllama-compat/wasm/…` | Worker + WASM compat per Safari (v3) |
| `autoFallback` | `true` | Usa i CDN web se gli asset locali mancano |
| `wllamaCdnVersion` | `3.13.0` | Versione di Wllama da usare come ripiego web |
| `modelFallbackUrl` | — | URL GGUF alternativo se il modello locale manca |
| `modelFallbackUrls` | — | Più URL GGUF di ripiego, provati in ordine |
| `systemPrompt` | assistente generico | Prompt di sistema predefinito |
| `contextSize` | `2048` | `n_ctx` (fissato al load) |
| `batchSize` | `256` | `n_batch` (fissato al load) |
| `threads` | metà dei core | Thread WASM (fissato al load) |
| `maxTokens` | `256` | Token massimi per risposta |
| `temperature` | `0.7` | Temperatura |
| `topK` / `topP` / `minP` | `40` / `0.9` / `0.05` | Campionamento |
| `repeatPenalty` | `1.1` | Penalità ripetizione |
| `seed` | `-1` | Seed (`-1` = casuale) |
| `chatTemplate` | `'chatml'` | `chatml`, `llama3`, `llama2`, `alpaca`, `plain` o funzione |
| `trimHistory` | `true` | Adatta la cronologia al contesto |
| `stopTokens` | dal template | Override token di stop |
| `allowOffline` | `false` | Consenti uso da cache offline |

### Metodi principali

| Metodo | Ritorna | Descrizione |
| --- | --- | --- |
| `load(opts?)` | `Promise<GGUFEngine>` | Scarica/carica il modello (idempotente) |
| `generate(prompt, opts?)` | `Promise<string>` | Una risposta a partire da un prompt |
| `ask(prompt, opts?)` | `Promise<string>` | Alias di `generate` |
| `generateDetailed(...)` | `Promise<Result>` | Come sopra + statistiche e prompt formattato |
| `stream(prompt, opts?)` | `AsyncIterable<string>` | Streaming dei delta di testo |
| `chat(messages, opts?)` | `Promise<string>` | Conversazione stateless |
| `chatStream(messages, opts?)` | `AsyncIterable<string>` | Streaming conversazione stateless |
| `send(message, opts?)` | `Promise<string>` | Conversazione con memoria interna |
| `sendStream(message, opts?)` | `AsyncIterable<string>` | Come `send` ma in streaming |
| `resetHistory(systemPrompt?)` | `this` | Azzera la cronologia |
| `abort()` | `boolean` | Interrompe la generazione in corso |
| `dispose()` / `unload()` | `Promise<void>` | Libera memoria/WASM |
| `configure(partial)` | `{ needsReload, options }` | Cambia i default |
| `on(event, handler)` | `() => void` | Registra un listener |
| `off(event, handler)` | — | Rimuove un listener |

Getters: `isLoaded`, `isBusy`, `status`, `lastResult`, `raw`, `rawGpu`,
`backend` (`'webgpu' | 'wasm' | null`), `isWebGPU`, `gpuInfo`, `backendLabel`,
`getModelInfo()`.

`Result`: `{ text, chunks, durationMs, tokensPerSecond, aborted, prompt,
backend, model }` (con `promptTokens` e `nPredict`).

### Opzioni per singola chiamata

Tutte le opzioni di generazione del costruttore (`maxTokens`, `temperature`, `topK`,
`topP`, `minP`, `repeatPenalty`, `seed`, `systemPrompt`, `chatTemplate`,
`stopTokens`) possono essere passate al singolo metodo, più:

- `onToken(delta, fullText)` — callback per ogni pezzo di testo;
- `signal` — un `AbortSignal` per annullare la richiesta.

---

## Esempi

### Streaming

```js
for await (const delta of llm.stream('Racconta una barzelletta')) {
  process.stdout.write(delta);
}
```

### Streaming con interruzione

```js
const controller = new AbortController();
setTimeout(() => controller.abort(), 3000);

for await (const delta of llm.stream('Scrivi un tema lunghissimo', {
  signal: controller.signal,
})) {
  console.log(delta);
}
```

### Chat multi-turn

```js
const risposta = await llm.chat([
  { role: 'system', content: 'Sei un traduttore italiano → inglese.' },
  { role: 'user', content: 'Buongiorno' },
  { role: 'assistant', content: 'Good morning' },
  { role: 'user', content: 'Come stai?' },
]);
```

### Memoria interna

```js
await llm.send('Mi chiamo Luca');
console.log(await llm.send('Come mi chiamo?')); // Luca
llm.resetHistory();
```

### Eventi

```js
llm.on('progress', ({ loaded, total, percent }) => {
  console.log(`${(loaded / total * 100).toFixed(1)}%`);
});

llm.on('token', ({ delta }) => console.log(delta));
llm.on('status', (s) => console.log('stato:', s)); // loading, ready, generating...
llm.on('fallback', ({ asset, from, to }) => console.log(asset, from, '→', to));
llm.on('error', (err) => console.error(err));
```

### Ripiego automatico su fonti web

All'avvio il motore verifica (HEAD/GET Range) se `vendor/wllama/esm/index.js`,
`vendor/wllama/esm/wasm/wllama.wasm` e gli asset compat sono raggiungibili:

- se ci sono, usa **sempre** gli asset locali del server;
- se mancano, importa runtime e WASM da `https://cdn.jsdelivr.net/npm/@wllama/wllama@3.13.0/esm/`;
- il compat (Safari) arriva da `@wllama/wllama-compat@3.13.0` solo se non è self-hosted;
- la scelta è per singolo file: se manca solo un `.wasm`, solo quello arriva dal CDN;
- se l'import del modulo locale fallisce comunque, si ritenta una volta sul CDN.

Anche il modello può avere un ripiego web:

```js
import { GGUFEngine, DEFAULT_MODEL_FALLBACK_URL } from './gguf-engine.js';

const llm = new GGUFEngine({
  modelUrl: './models/mio-modello.gguf',
  modelFallbackUrl: DEFAULT_MODEL_FALLBACK_URL, // usato solo se il locale fallisce
});
```

Per disattivare ogni ripiego: `autoFallback: false`. Per forzare il CDN: passa
`wllamaModuleUrl`/`wasmPaths` con URL `https://…`. Se aggiungi o rimuovi asset a
runtime, chiama `GGUFEngine.invalidateAssetCache()`.

Getter utili dopo il load: `activeModelUrl` e `runtimeSource` (`'local'`, `'cdn'`
o `'custom'`).

### Template personalizzato

```js
const llm = new GGUFEngine({
  modelUrl: '...',
  chatTemplate: (messages) => ({
    prompt: messages
      .map((m) => `### ${m.role}:\n${m.content}`)
      .join('\n\n') + '\n\n### assistant:\n',
    stopTokens: ['### user:', '### system:'],
  }),
});
```

### WASM self-hosted (offline / senza CDN)

Basta eseguire `bun run deps`: gli asset locali, quando presenti, hanno la
precedenza. I path restano comunque sovrascrivibili:

```js
const llm = new GGUFEngine({
  modelUrl: './models/mio-modello.gguf',
  wllamaModuleUrl: './vendor/wllama/esm/index.js',
  wasmPaths: {
    default: './vendor/wllama/esm/wasm/wllama.wasm',
  },
  wllamaCompatPaths: {
    worker: './vendor/wllama-compat/wasm/wllama.js',
    wasm: './vendor/wllama-compat/wasm/wllama.wasm',
  },
});
```

---

## Backend WebGPU (accelerazione GPU)

Con `backend: 'auto'` (default) il motore prova prima la GPU:

```js
const llm = new GGUFEngine({
  backend: 'auto',
  // Id del modello MLC gestito da WebLLM (non è un file GGUF).
  webgpuModel: 'LFM2.5-350M-q4f32_1-MLC',
  // Fallback CPU: usato se WebGPU non c'è o il modello GPU non si carica.
  modelUrl: './models/MNN-Pocket-0.3B-Instruct-Q4_1.gguf',
  webgpuFallbackToWasm: true,
});

await llm.load();
console.log(llm.backendLabel);   // es. "WebGPU · NVIDIA GeForce RTX 4070"
console.log(await llm.generate('Ciao!'));
```

Note:

- i modelli GPU sono **MLC**, identificati da un id noto a WebLLM
  (`prebuiltAppConfig.model_list`); per usarne uno non precompilato passa
  `webgpuModelRecord` con il record completo;
- il modello viene scaricato la prima volta e poi resta nella cache del browser
  (Cache Storage / OPFS), quindi i caricamenti successivi sono immediati;
- `webgpuContextSize` (default `4096`) viene usato per il trimming della
  cronologia; per contesti più grandi serve VRAM proporzionale;
- per forzare la GPU: `backend: 'webgpu'` — in questo caso, se WebGPU manca,
  `load()` lancia un `WebGPULoadError` invece di ripiegare;
- per escluderla: `backend: 'wasm'`;
- `llm.rawGpu` espone l'engine WebLLM, `llm.gpuInfo` le informazioni
  sull'adattatore (`vendor`, `architecture`, `description`, `features`).

Utility esportate (utili per una UI di diagnostica):

```js
import { WebGPUBackend, describeAdapter } from './gguf-engine.js';

const gpu = await WebGPUBackend.detect();
if (gpu.supported) {
  console.log(`GPU: ${describeAdapter(gpu.adapterInfo)}`);
} else {
  console.log(`WebGPU non disponibile: ${gpu.reason}`);
}
```

Requisiti: browser con WebGPU (Chrome/Edge 113+, Safari 26+, Firefox con
`dom.webgpu.enabled`), pagina in contesto sicuro (`https://` o `localhost`).
Per l'import del runtime non servono header speciali; per i thread WASM sì
(vedi sotto).

---

## Compatibilità dei modelli (architetture)

Wllama include un build di `llama.cpp`: un GGUF la cui architettura è più
recente del build **non può** essere caricato, anche se il file è integro.
Il sintomo tipico è un errore fuorviante:

```
Invalid typed array length: 1163217991
```

Non è un problema di RAM né di download: `1163217991` (0x45554C47) sono i byte
ASCII `GLUE`, cioè l'intestazione del protocollo interno di Wllama. Quando
`llama.cpp` rifiuta il modello, il worker non scrive la lunghezza dell'output e
Wllama legge quei 4 byte come lunghezza, provando ad allocare ~1,1 GB.
L'errore vero di `llama.cpp` viene perso.

Versioni di riferimento:

| Wllama | Data | Supporto `lfm2` (LFM2/LFM2.5) |
| --- | --- | --- |
| `2.3.2` | giu 2025 | no |
| `>= 2.3.3` | lug 2025 | sì |
| `>= 3.13.0` | 2026 | sì, API chat/completion OpenAI-compatibile (consigliata) |

Per evitare il problema:

1. tieni allineati `WLLAMA_CDN_VERSION` in `gguf-engine.js` e i default di
   `scripts/download-deps.js` / `scripts/download-deps.sh`;
2. dopo un aggiornamento riscarica gli asset locali con
   `bun run deps --force` (il `vendor/` locale ha la precedenza sul CDN);
3. svuota la cache OPFS del browser (`Wllama.cacheManager.clear()` o
   cancellando i dati del sito) se il modello risultava già scaricato;
4. verifica l'architettura reale del GGUF (campo `general.architecture`):
   `lfm2` richiede un runtime aggiornato; `llama`, `qwen2`, `gemma3`, ecc.
   funzionano anche con build più vecchi.
5. `llm.libllamaVersion` espone la versione di `llama.cpp` del runtime in uso
   (es. `b12211-7d7f028` con Wllama 3.13.0); se è `null`, il runtime è troppo
   vecchio per qualunque architettura recente.

Con Wllama 3.x il motore usa l'API OpenAI-compatibile nativa
(`createChatCompletion` / `createCompletion`) e `setCompat` per Safari; non
dipendono più da `tokenize`, `detokenize`, `formatChat` o dai token EOG
numerici usati dalle versioni 2.x.

---

## Note importanti

- **Multi-thread WASM**: richiede `SharedArrayBuffer`, quindi la pagina deve essere
  servita con `Cross-Origin-Opener-Policy: same-origin` e
  `Cross-Origin-Embedder-Policy: require-corp` (il `server.js` di questo progetto
  lo fa già). Senza questi header viene usato automaticamente 1 thread.
- **WebGPU**: richiede un contesto sicuro e un browser con supporto WebGPU. Se
  manca, con `backend: 'auto'` il motore ripiega su WASM e lo segnala con
  l'evento `fallback` (`{ asset: 'backend', from: 'webgpu', to: 'wasm', reason }`).
- **VRAM**: i modelli GPU occupano memoria video; un modello troppo grande per
  la GPU fallisce il caricamento e, con `webgpuFallbackToWasm`, si passa al GGUF.
- **Cache**: Wllama usa la Cache API del browser; WebLLM usa Cache Storage/OPFS.
  I caricamenti successivi sono molto più rapidi.
- **CORS**: con gli asset locali non serve alcuna configurazione CORS. Se usi un
  modello su un altro dominio, quel server deve consentire richieste cross-origin.
  I CDN di ripiego (jsDelivr, Hugging Face) inviano già gli header necessari.
- **Cache**: Wllama usa la Cache API del browser; i caricamenti successivi sono
  molto più rapidi.
- **Memoria**: `contextSize` e `batchSize` alti consumano più RAM. Se il browser va
  in OOM, riducili o usa una quantizzazione più piccola (es. Q4).
- `contextSize`, `batchSize` e `threads` vengono fissati al caricamento: per
  cambiarli bisogna creare una nuova istanza (o `dispose()` + nuova istanza).
