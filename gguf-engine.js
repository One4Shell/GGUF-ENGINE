/**
 * ============================================================================
 * GGUFEngine — classe riutilizzabile per modelli GGUF nel browser
 * ============================================================================
 * Wrapper di alto livello attorno a @wllama/wllama.
 *
 * Obiettivo: passi l'URL del modello GGUF e ottieni subito un'API semplice,
 * con streaming, abort, cronologia multi-turn e template di chat.
 *
 * Uso minimo:
 *
 *   import { GGUFEngine } from './gguf-engine.js';
 *
 *   const llm = new GGUFEngine({
 *     modelUrl: './models/MNN-Pocket-0.3B-Instruct-Q4_1.gguf',
 *     systemPrompt: 'Sei un assistente utile e conciso.',
 *   });
 *
 *   // "bun run deps" è opzionale: se vendor/ e models/ non ci sono, il motore
 *   // usa automaticamente runtime e modello dalle fonti web.
 *
 *   await llm.load();                       // scarica/carica in cache
 *   const risposta = await llm.generate('Ciao, chi sei?');
 *   console.log(risposta);
 *
 * Streaming:
 *
 *   for await (const delta of llm.stream('Racconta una barzelletta')) {
 *     process.stdout.write(delta);
 *   }
 *
 * Chat multi-turn (stateless):
 *
 *   const out = await llm.chat([
 *     { role: 'system', content: 'Sei un traduttore IT→EN.' },
 *     { role: 'user', content: 'Ciao mondo' },
 *   ]);
 *
 * Chat multi-turn con memoria interna:
 *
 *   await llm.send('Mi chiamo Luca');
 *   console.log(await llm.send('Come mi chiamo?')); // "Luca"
 *   llm.resetHistory();
 *
 * Attenzione: il multi-thread WASM richiede SharedArrayBuffer, quindi la pagina
 * deve essere servita con gli header COOP/COEP (come fa server.js di questo
 * progetto). In caso contrario il motore usa automaticamente 1 thread.
 * ============================================================================
 */

import {
  WebGPUBackend,
  DEFAULT_WEBLLM_MODEL,
  DEFAULT_WEBLLM_MODULE_URL,
  describeAdapter,
  gpuError,
} from './webgpu-backend.js';

// Runtime e binari sono normalmente serviti localmente (vedi scripts/download-deps.js).
// Se il server non espone gli asset in "vendor", il motore ripiega da solo sui
// CDN web: l'app continua a funzionare, senza errori. Quando i file locali sono
// presenti hanno sempre la precedenza.
const DEFAULT_WLLAMA_MODULE = './vendor/wllama/esm/index.js';

// Wllama 3.x: un solo WASM principale (JSPI/MEMORY64). I path legacy
// single-thread/multi-thread restano accettati e vengono normalizzati.
const DEFAULT_WASM_PATHS = {
  default: './vendor/wllama/esm/wasm/wllama.wasm',
};

// Worker + WASM "compat" per browser senza JSPI/MEMORY64 (Safari, Firefox
// senza flag). Wllama li cerca da solo, ma senza override userebbe il CDN.
const DEFAULT_WLLAMA_COMPAT_PATHS = {
  worker: './vendor/wllama-compat/wasm/wllama.js',
  wasm: './vendor/wllama-compat/wasm/wllama.wasm',
};

// Runtime WebLLM self-hosted (opzionale, vedi `bun run deps --with-webllm`).
const DEFAULT_WEBLLM_VENDOR_MODULE = './vendor/web-llm/lib/index.js';

/**
 * Versione del pacchetto usata come ripiego web (allineata a download-deps.js).
 *
 * Wllama 3.x include un llama.cpp recente (JSPI/MEMORY64, WebGPU, API
 * OpenAI-compatibile). Tenere questa versione allineata al runtime vendor/
 * scaricato con `bun run deps`.
 */
const WLLAMA_CDN_VERSION = '3.13.0';

/** Fonti web di ripiego per runtime e binari WASM. */
const CDN_WLLAMA_URLS = {
  version: WLLAMA_CDN_VERSION,
  moduleUrl: `https://cdn.jsdelivr.net/npm/@wllama/wllama@${WLLAMA_CDN_VERSION}/esm/index.js`,
  wasmPaths: {
    default: `https://cdn.jsdelivr.net/npm/@wllama/wllama@${WLLAMA_CDN_VERSION}/esm/wasm/wllama.wasm`,
  },
  compat: {
    worker: `https://cdn.jsdelivr.net/npm/@wllama/wllama-compat@${WLLAMA_CDN_VERSION}/wasm/wllama.js`,
    wasm: `https://cdn.jsdelivr.net/npm/@wllama/wllama-compat@${WLLAMA_CDN_VERSION}/wasm/wllama.wasm`,
  },
};

/** Modello remoto usato come ripiego opzionale se il GGUF locale non esiste. */
export const DEFAULT_MODEL_FALLBACK_URL =
  'https://huggingface.co/taobao-mnn/MNN-Pocket-0.3B-GGUF/resolve/main/MNN-Pocket-0.3B-Instruct-Q4_1.gguf';

/**
 * Token di stop noti per ogni famiglia di template.
 * Servono solo come punto di partenza: i marker presenti nel template GGUF
 * vengono analizzati e aggiunti automaticamente (vedi detectStopStrings).
 */
const FAMILY_STOP_TOKENS = {
  chatml: ['<|im_end|>', '<|endoftext|>'],
  llama3: ['<|eot_id|>', '<|end_of_text|>', '<|eom_id|>'],
  llama2: ['</s>', '[/INST]'],
  alpaca: ['</s>', '### Instruction:', '### Response:'],
  plain: [],
  gemma: ['<end_of_turn>', '<eos>'],
  phi3: ['<|end|>', '<|endoftext|>'],
  commandr: ['<|END_OF_TURN_TOKEN|>'],
  chatglm: ['<|user|>', '<|observation|>'],
  minicpm: ['<|im_end|>', '</s>'],
  deepseek: [
    '<|end▁of▁sentence|>',
    '<｜end▁of▁sentence｜>',
    '<|end_of_sentence|>',
    '<|EOT|>',
    '<|endoftext|>',
  ],
  zephyr: ['</s>', '<|endoftext|>'],
  openchat: ['<|end_of_turn|>'],
  falcon: ['<|endoftext|>'],
  vicuna: ['</s>', 'USER:'],
  granite: ['<|end_of_text|>'],
};

/** Opzioni di default della classe. */
const DEFAULT_OPTIONS = {
  // Backend
  backend: 'auto', // 'auto' | 'wasm' (Wllama/CPU) | 'webgpu' (WebLLM/GPU)
  webgpuModel: null, // id MLC, es. 'LFM2.5-350M-q4f32_1-MLC'
  webgpuModelRecord: null, // ModelRecord custom per self-hosting
  webllmModuleUrl: null, // null = CDN @mlc-ai/web-llm
  webgpuContextSize: 4096, // usato solo dal backend GPU (trimming cronologia)
  webgpuFallbackToWasm: true, // con backend 'auto': ripiega su GGUF/WASM

  // Caricamento
  contextSize: 'auto', // 'auto' | 'max' | numero
  batchSize: 'auto', // 'auto' | numero
  threads: null, // null = metà dei core disponibili (se multi-thread attivo)
  parallelDownloads: 3,

  // Rilevamento automatico dal file GGUF
  autoDetect: true, // metadati GGUF, template di chat e token di stop
  preflightMetadata: true, // piccola richiesta Range per leggere n_ctx_train

  // Generazione
  systemPrompt: '',
  maxTokens: 256,
  temperature: 0.7,
  topK: 40,
  topP: 0.9,
  minP: 0.05,
  repeatPenalty: 1.1,
  seed: -1,

  // Comportamento
  chatTemplate: 'auto', // 'auto' | chatml | llama3 | llama2 | gemma | phi3 | ... | funzione
  trimHistory: true,
  stopTokens: null, // null = rilevati automaticamente dal modello
};

/* ---------------------------------------------------------------------------
 * Utility interne
 * ------------------------------------------------------------------------- */

function clampInt(value, min, max) {
  const n = Number.parseInt(value, 10);
  if (!Number.isFinite(n)) return min;
  return Math.min(max, Math.max(min, n));
}

/** Stima grezza dei token: ~3.5 caratteri per token. */
function estimateTokens(text) {
  return Math.ceil(String(text ?? '').length / 3.5);
}

/**
 * Normalizza l'URL del modello. Supporta anche la forma abbreviata Hugging Face:
 *
 *   hf:owner/repo/file.gguf       → https://huggingface.co/owner/repo/resolve/main/file.gguf
 *   hf://owner/repo/file.gguf     → idem
 */
function normalizeModelUrl(url) {
  if (typeof url !== 'string') return url;
  const trimmed = url.trim();
  const match = /^hf:(?:\/\/)?(.+)$/i.exec(trimmed);
  if (!match) return trimmed;
  const parts = match[1].split('/').filter(Boolean);
  if (parts.length < 3) return trimmed;
  const [owner, repo, ...rest] = parts;
  return `https://huggingface.co/${owner}/${repo}/resolve/main/${rest.join('/')}`;
}

/** Token speciali che non devono mai comparire nel testo restituito. */
const SPECIAL_TOKENS_RE =
  /<\|(?:im_start|im_end|im_sep|start_header_id|end_header_id|eot_id|eom_id|begin_of_text|end_of_text|endoftext|file_separator|start_of_role|end_of_role|end|end_of_turn|start_of_turn|system|user|assistant|observation|tool|tool_call|tool_response)\|>|<\/s>|<s>|<eos>|<bos>|<pad>|<end_of_turn>|<start_of_turn>|<|im_end|>|<|im_start|>/g;

/** Rimuove i token speciali e i marker di stop dal testo generato. */
function stripSpecialTokens(raw, stopStrings = []) {
  if (!raw) return '';
  let out = String(raw);
  // I marker di stop (anche multi-token) delimitano la fine della risposta.
  let cut = -1;
  for (const stop of stopStrings) {
    if (!stop) continue;
    const index = out.indexOf(stop);
    if (index !== -1 && (cut === -1 || index < cut)) cut = index;
  }
  if (cut !== -1) out = out.slice(0, cut);
  return out.replace(SPECIAL_TOKENS_RE, '');
}

/** Indice del primo marker di stop presente nel testo, oppure -1. */
function findStopIndex(text, stopStrings) {
  if (!text || !stopStrings?.length) return -1;
  let cut = -1;
  for (const stop of stopStrings) {
    if (!stop) continue;
    const index = text.indexOf(stop);
    if (index !== -1 && (cut === -1 || index < cut)) cut = index;
  }
  return cut;
}

/** Lunghezza del prefisso comune tra due stringhe. */
function commonPrefixLength(a, b) {
  const max = Math.min(a.length, b.length);
  let i = 0;
  while (i < max && a[i] === b[i]) i += 1;
  return i;
}

/**
 * Converte un URL relativo in assoluto.
 *
 * Necessario perché il download del modello avviene dentro una Web Worker
 * creata da un blob URL: lì `fetch()` non riesce a risolvere i path relativi
 * ("Failed to parse URL from ./models/...").
 */
function toAbsoluteUrl(url, base) {
  if (typeof url !== 'string') return url;
  // URL già assoluto (https:, blob:, data:, file:, //cdn...) → lascia intatto
  if (/^[a-z][a-z0-9+.-]*:/i.test(url) || url.startsWith('//')) return url;
  const fallbackBase =
    globalThis.document?.baseURI ||
    globalThis.location?.href ||
    import.meta.url;
  try {
    return new URL(url, base || fallbackBase).href;
  } catch {
    return url;
  }
}

/** Cache delle verifiche di esistenza (una sola richiesta per URL). */
const assetAvailability = new Map();

/**
 * Verifica se un asset è raggiungibile sul server.
 * Usa HEAD e, se il server non lo supporta, una GET con Range minimo.
 * Per URL non HTTP (blob:, data:, file:) non è possibile verificare: li diamo
 * per disponibili e lasciamo che l'import/il caricamento decida.
 */
async function assetExists(url) {
  const target = toAbsoluteUrl(url);
  if (!/^https?:/i.test(target)) return true;
  if (assetAvailability.has(target)) return assetAvailability.get(target);

  const check = (async () => {
    const attempts = [
      { method: 'HEAD' },
      { method: 'GET', headers: { Range: 'bytes=0-0' } },
    ];
    for (const init of attempts) {
      let response = null;
      try {
        response = await fetch(target, { ...init, cache: 'no-store' });
        const available =
          response.ok || response.status === 206 || response.status === 304;
        if (available) {
          // Alcuni server SPA rispondono 200 con index.html per i file mancanti:
          // un asset JS/WASM non può essere HTML, quindi lo consideriamo assente.
          const contentType = (
            response.headers.get('content-type') || ''
          ).toLowerCase();
          return !contentType.includes('text/html');
        }
        // 404/410 sono definitivi; 405/501 (HEAD non supportato) e altri
        // errori transitori passano al tentativo successivo.
        if (response.status === 404 || response.status === 410) return false;
      } catch {
        /* tenta il metodo successivo */
      } finally {
        // Se il server ignora Range, evitiamo di scaricare tutto l'asset.
        try {
          await response?.body?.cancel();
        } catch {
          /* ignora */
        }
      }
    }
    return false;
  })();

  assetAvailability.set(target, check);
  return check;
}

/** Wllama 3.x ha un layout e un'API diversi da 2.x. */
function isWllamaV3(version) {
  const major = Number.parseInt(String(version || WLLAMA_CDN_VERSION).split('.')[0], 10);
  return !Number.isFinite(major) || major >= 3;
}

/** Accetta sia la config v3 ({default}) sia le mappe legacy 2.x. */
function normalizeWasmPaths(paths) {
  const normalized = { ...(paths || {}) };
  if (!normalized.default) {
    normalized.default =
      normalized['multi-thread/wllama.wasm'] ||
      normalized['single-thread/wllama.wasm'] ||
      DEFAULT_WASM_PATHS.default;
  }
  return normalized;
}

/** Fonti CDN per una specifica versione di Wllama. */
function cdnRuntimeSources(version) {
  if (!version || version === WLLAMA_CDN_VERSION) {
    return {
      version: CDN_WLLAMA_URLS.version,
      moduleUrl: CDN_WLLAMA_URLS.moduleUrl,
      wasmPaths: { ...CDN_WLLAMA_URLS.wasmPaths },
      compat: { ...CDN_WLLAMA_URLS.compat },
    };
  }

  const encoded = encodeURIComponent(version);
  const base = `https://cdn.jsdelivr.net/npm/@wllama/wllama@${encoded}/esm`;
  if (!isWllamaV3(version)) {
    return {
      version,
      moduleUrl: `${base}/index.js`,
      wasmPaths: {
        'single-thread/wllama.wasm': `${base}/single-thread/wllama.wasm`,
        'multi-thread/wllama.wasm': `${base}/multi-thread/wllama.wasm`,
      },
      compat: null,
    };
  }

  return {
    version,
    moduleUrl: `${base}/index.js`,
    wasmPaths: { default: `${base}/wasm/wllama.wasm` },
    compat: {
      worker: `https://cdn.jsdelivr.net/npm/@wllama/wllama-compat@${encoded}/wasm/wllama.js`,
      wasm: `https://cdn.jsdelivr.net/npm/@wllama/wllama-compat@${encoded}/wasm/wllama.wasm`,
    },
  };
}

/** Risoluzioni già calcolate, per versione del runtime. */
const runtimeSourcesCache = new Map();

/**
 * Sceglie runtime e WASM locali se disponibili, altrimenti le fonti web.
 * La verifica è "per file": un singolo asset mancante non costringe a usare
 * il CDN per tutto il resto.
 */
function resolveDefaultRuntimeSources(version) {
  const key = version || WLLAMA_CDN_VERSION;
  if (runtimeSourcesCache.has(key)) return runtimeSourcesCache.get(key);

  const resolution = (async () => {
    const cdn = cdnRuntimeSources(version);
    const localModuleUrl = toAbsoluteUrl(DEFAULT_WLLAMA_MODULE, import.meta.url);

    // Verifica in parallelo per non sommare le latenze degli asset.
    const [hasLocalModule, wasmAvailability, compatAvailability] = await Promise.all([
      assetExists(localModuleUrl),
      Promise.all(
        Object.entries(DEFAULT_WASM_PATHS).map(async ([name, localPath]) => {
          const localUrl = toAbsoluteUrl(localPath, import.meta.url);
          return [name, localUrl, await assetExists(localUrl)];
        })
      ),
      Promise.all(
        Object.entries(DEFAULT_WLLAMA_COMPAT_PATHS).map(async ([name, localPath]) => {
          const localUrl = toAbsoluteUrl(localPath, import.meta.url);
          return [name, localUrl, await assetExists(localUrl)];
        })
      ),
    ]);

    const wasmPaths = {};
    for (const [name, localUrl, available] of wasmAvailability) {
      if (available) wasmPaths[name] = localUrl;
      else if (cdn.wasmPaths[name]) wasmPaths[name] = cdn.wasmPaths[name];
    }
    // Per versioni 2.x i path CDN sono diversi: aggiungili se mancanti.
    for (const [name, url] of Object.entries(cdn.wasmPaths)) {
      if (!(name in wasmPaths)) wasmPaths[name] = url;
    }

    let compat = cdn.compat ? { ...cdn.compat } : null;
    if (compat) {
      for (const [name, localUrl, available] of compatAvailability) {
        if (available) compat[name] = localUrl;
      }
    }

    return {
      moduleUrl: hasLocalModule ? localModuleUrl : cdn.moduleUrl,
      wasmPaths,
      compat,
      source: hasLocalModule ? 'local' : 'cdn',
    };
  })();

  runtimeSourcesCache.set(key, resolution);
  return resolution;
}

function hardwareThreads() {
  const cores = globalThis.navigator?.hardwareConcurrency;
  return Math.max(1, Number.isFinite(cores) ? cores : 4);
}

function multiThreadAvailable() {
  try {
    return (
      globalThis.crossOriginIsolated === true &&
      typeof SharedArrayBuffer !== 'undefined'
    );
  } catch {
    return false;
  }
}

function friendlyLoadError(err, context = {}) {
  const raw = err?.message || String(err || 'Errore sconosciuto.');
  const { architecture = null, libVersion = null } = context;
  let message = raw;

  if (/failed to fetch|networkerror|load failed/i.test(raw)) {
    message =
      'Impossibile scaricare il modello. Verifica la connessione, i blocchi CORS/firewall e che l\'URL GGUF sia raggiungibile.';
  } else if (/\bOOM\b|out of memory|memory allocation|not enough memory|insufficient memory/i.test(raw)) {
    message =
      'Memoria WebAssembly insufficiente. Chiudi altre schede, riduci contextSize/batchSize oppure usa una quantizzazione più piccola.';
  } else if (/404|not found/i.test(raw)) {
    message = `Modello non trovato (404). Controlla l'URL: ${raw}`;
  } else if (/failed to parse url|invalid url/i.test(raw)) {
    message = `URL del modello non valido o relativo: le Web Worker non risolvono i path relativi. Usa un URL assoluto (es. new URL('./models/modello.gguf', document.baseURI).href). Dettaglio: ${raw}`;
  } else if (/n_ctx|context size|create context|context creation/i.test(raw)) {
    message =
      'Impossibile creare il contesto del modello. Riduci contextSize (o lascialo su "auto") e riprova.';
  } else if (/unsupported model architecture|unknown architecture|not supported/i.test(raw)) {
    message = `Architettura GGUF non supportata da questa versione di llama.cpp/Wllama: ${raw}`;
  } else if (/invalid typed array length|invalid array length/i.test(raw)) {
    // Bug di Wllama <= 2.3.7: quando llama.cpp rifiuta il modello (architettura
    // non supportata, file/cache corrotti) il worker non scrive la lunghezza
    // dell'output e legge i primi 4 byte del messaggio GLUE ("GLUE" =
    // 0x45554C47 = 1163217991) come lunghezza, quindi prova ad allocare ~1,1 GB
    // e fallisce con questo RangeError. L'errore reale di llama.cpp va perso.
    const details = [
      architecture ? `architettura rilevata: "${architecture}"` : null,
      libVersion ? `llama.cpp: ${libVersion}` : 'versione di llama.cpp sconosciuta (runtime Wllama molto vecchio)',
    ].filter(Boolean).join(', ');
    message =
      'Wllama/llama.cpp non è riuscito a caricare questo modello. ' +
      'Il messaggio "Invalid typed array length" è un bug di Wllama che nasconde ' +
      'l\'errore reale di llama.cpp: di solito significa architettura non supportata ' +
      'dalla versione di llama.cpp inclusa, oppure GGUF/cache corrotti. ' +
      `(${details}) ` +
      'Aggiorna @wllama/wllama ad almeno 3.13.0, riscarica gli asset ' +
      'con "bun run deps --force" e svuota la cache OPFS, oppure usa un modello di ' +
      'un\'architettura già supportata.';
  }

  const wrapped = new Error(message);
  wrapped.name = 'GGUFLoadError';
  wrapped.cause = err;
  return wrapped;
}

/** Riconosce gli errori di memoria esaurita (per il retry con contesto ridotto). */
function isOutOfMemoryError(err) {
  const raw = `${err?.message || err || ''}`;
  return /out of memory|\boom\b|memory allocation|insufficient memory|not enough memory|failed to allocate|cannot allocate/i.test(
    raw
  );
}

/* ---------------------------------------------------------------------------
 * Template di chat
 * ------------------------------------------------------------------------- */

function buildChatML(messages) {
  let out = '';
  for (const msg of messages) {
    const role = msg.role || 'user';
    out += `<|im_start|>${role}\n${String(msg.content ?? '').trim()}<|im_end|>\n`;
  }
  out += '<|im_start|>assistant\n';
  return out;
}

function buildLlama3(messages) {
  let out = '';
  for (const msg of messages) {
    const role = msg.role || 'user';
    out += `<|start_header_id|>${role}<|end_header_id|>\n\n${String(
      msg.content ?? ''
    ).trim()}<|eot_id|>`;
  }
  out += '<|start_header_id|>assistant<|end_header_id|>\n\n';
  return out;
}

function buildLlama2(messages) {
  const system = messages
    .filter((m) => m.role === 'system')
    .map((m) => String(m.content ?? '').trim())
    .filter(Boolean)
    .join('\n\n');

  const turns = messages.filter((m) => m.role !== 'system');
  let out = '<s>';
  let firstUser = true;

  for (const msg of turns) {
    const content = String(msg.content ?? '').trim();
    if (msg.role === 'assistant') {
      out += ` ${content} </s><s>`;
      continue;
    }

    const sysBlock = firstUser && system ? `<<SYS>>\n${system}\n<</SYS>>\n\n` : '';
    out += `[INST] ${sysBlock}${content} [/INST]`;
    firstUser = false;
  }

  return out;
}

function buildAlpaca(messages) {
  const system = messages
    .filter((m) => m.role === 'system')
    .map((m) => String(m.content ?? '').trim())
    .filter(Boolean)
    .join('\n\n');
  const rows = messages
    .filter((m) => m.role !== 'system')
    .map((m) => {
      const label = m.role === 'assistant' ? '### Response:' : '### Instruction:';
      return `${label}\n${String(m.content ?? '').trim()}`;
    });

  let out = system ? `${system}\n\n` : '';
  out += rows.join('\n\n');
  if (messages.at(-1)?.role !== 'assistant') out += '\n\n### Response:\n';
  return out;
}

function buildPlain(messages) {
  const out = messages
    .filter((m) => m.role !== 'system')
    .map((m) => String(m.content ?? '').trim())
    .join('\n\n');
  return messages.at(-1)?.role === 'assistant' ? out : `${out}\n\nAssistant:`;
}

function buildGemma(messages) {
  const system = messages
    .filter((m) => m.role === 'system')
    .map((m) => String(m.content ?? '').trim())
    .filter(Boolean)
    .join('\n\n');
  const turns = messages.filter((m) => m.role !== 'system');

  let out = '';
  let firstUser = true;
  for (const msg of turns) {
    const content = String(msg.content ?? '').trim();
    const role = msg.role === 'assistant' ? 'model' : 'user';
    const prefix = firstUser && system ? `${system}\n\n` : '';
    out += `<start_of_turn>${role}\n${prefix}${content}<end_of_turn>\n`;
    firstUser = false;
  }
  if (messages.at(-1)?.role !== 'assistant') out += '<start_of_turn>model\n';
  return out;
}

function buildPhi3(messages) {
  let out = '';
  for (const msg of messages) {
    const content = String(msg.content ?? '').trim();
    if (!content) continue;
    const role = msg.role === 'assistant' ? 'assistant' : msg.role === 'system' ? 'system' : 'user';
    out += `<|${role}|>\n${content}<|end|>\n`;
  }
  out += '<|assistant|>\n';
  return out;
}

function buildCommandR(messages) {
  let out = '';
  for (const msg of messages) {
    const content = String(msg.content ?? '').trim();
    if (!content) continue;
    const role = msg.role === 'assistant' ? 'CHATBOT' : msg.role === 'system' ? 'SYSTEM' : 'USER';
    out += `<|START_OF_TURN_TOKEN|><|${role}_TOKEN|>${content}<|END_OF_TURN_TOKEN|>`;
  }
  out += '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>';
  return out;
}

function buildChatGLM(messages) {
  let out = '[gMASK]<sop>';
  for (const msg of messages) {
    const content = String(msg.content ?? '').trim();
    if (!content) continue;
    const role = msg.role === 'assistant' ? 'assistant' : msg.role === 'system' ? 'system' : 'user';
    out += `<|${role}|>\n${content}`;
  }
  if (messages.at(-1)?.role !== 'assistant') out += '<|assistant|>\n';
  return out;
}

function buildDeepSeek(messages) {
  const system = messages
    .filter((m) => m.role === 'system')
    .map((m) => String(m.content ?? '').trim())
    .filter(Boolean)
    .join('\n\n');
  const turns = messages.filter((m) => m.role !== 'system');

  let out = `<|begin\u2581of\u2581sentence|>${system}`;
  for (const msg of turns) {
    const content = String(msg.content ?? '').trim();
    if (!content) continue;
    if (msg.role === 'assistant') {
      out += `<|Assistant|>${content}<|end\u2581of\u2581sentence|>`;
    } else {
      out += `<|User|>${content}`;
    }
  }
  if (messages.at(-1)?.role !== 'assistant') out += '<|Assistant|>';
  return out;
}

function buildZephyr(messages) {
  let out = '';
  for (const msg of messages) {
    const content = String(msg.content ?? '').trim();
    if (!content) continue;
    const role = msg.role === 'assistant' ? 'assistant' : msg.role === 'system' ? 'system' : 'user';
    out += `<|${role}|>\n${content}</s>\n`;
  }
  if (messages.at(-1)?.role !== 'assistant') out += '<|assistant|>';
  return out;
}

function buildOpenChat(messages) {
  let out = '';
  for (const msg of messages) {
    const content = String(msg.content ?? '').trim();
    if (!content) continue;
    const role = msg.role === 'assistant' ? 'Assistant' : msg.role === 'system' ? 'System' : 'User';
    out += `GPT4 Correct ${role}: ${content}<|end_of_turn|>`;
  }
  out += 'GPT4 Correct Assistant:';
  return out;
}

function buildVicuna(messages) {
  const system = messages
    .filter((m) => m.role === 'system')
    .map((m) => String(m.content ?? '').trim())
    .filter(Boolean)
    .join('\n\n');
  const turns = messages.filter((m) => m.role !== 'system');

  let out = system ? `${system}\n\n` : '';
  for (const msg of turns) {
    const content = String(msg.content ?? '').trim();
    if (!content) continue;
    out += msg.role === 'assistant' ? `${content}</s>` : `USER: ${content}\n`;
  }
  if (messages.at(-1)?.role !== 'assistant') out += 'ASSISTANT:';
  return out;
}

function buildFalcon(messages) {
  const system = messages
    .filter((m) => m.role === 'system')
    .map((m) => String(m.content ?? '').trim())
    .filter(Boolean)
    .join('\n\n');
  const turns = messages.filter((m) => m.role !== 'system');

  let out = system ? `${system}\n\n` : '';
  for (const msg of turns) {
    const content = String(msg.content ?? '').trim();
    if (!content) continue;
    const label = msg.role === 'assistant' ? 'Assistant' : 'User';
    out += `${label}: ${content}\n\n`;
  }
  if (messages.at(-1)?.role !== 'assistant') out += 'Assistant:';
  return out;
}

function buildGranite(messages) {
  let out = '';
  for (const msg of messages) {
    const content = String(msg.content ?? '').trim();
    if (!content) continue;
    const role = msg.role === 'assistant' ? 'assistant' : msg.role === 'system' ? 'system' : 'user';
    out += `<|start_of_role|>${role}<|end_of_role|>${content}<|end_of_text|>\n`;
  }
  if (messages.at(-1)?.role !== 'assistant') {
    out += '<|start_of_role|>assistant<|end_of_role|>';
  }
  return out;
}

function buildMiniCPM(messages) {
  let out = '';
  for (const msg of messages) {
    const content = String(msg.content ?? '').trim();
    if (!content) continue;
    if (msg.role === 'assistant') out += `<AI>${content}`;
    else out += `<\u7528\u6237>${content}`;
  }
  if (messages.at(-1)?.role !== 'assistant') out += '<AI>';
  return out;
}

const TEMPLATE_BUILDERS = {
  chatml: buildChatML,
  llama3: buildLlama3,
  llama2: buildLlama2,
  alpaca: buildAlpaca,
  plain: buildPlain,
  gemma: buildGemma,
  phi3: buildPhi3,
  commandr: buildCommandR,
  chatglm: buildChatGLM,
  deepseek: buildDeepSeek,
  zephyr: buildZephyr,
  openchat: buildOpenChat,
  vicuna: buildVicuna,
  falcon: buildFalcon,
  granite: buildGranite,
  minicpm: buildMiniCPM,
};

/**
 * Marker che identificano la famiglia di template a partire dal testo del
 * template Jinja contenuto nel GGUF. L'ordine conta: i marker più specifici
 * vengono prima di quelli generici (es. chatml).
 */
const TEMPLATE_MARKERS = [
  ['deepseek', [/[｜│]|▁of▁sentence/i]],
  ['granite', [/start_of_role/i]],
  ['llama3', [/start_header_id|eot_id|eom_id|begin_of_text/i]],
  ['commandr', [/START_OF_TURN_TOKEN/i]],
  ['gemma', [/start_of_turn/]],
  ['phi3', [/<\|end\|>/i]],
  ['chatglm', [/\[gMASK\]|<sop>/i]],
  ['minicpm', [/<\u7528\u6237>/]],
  ['openchat', [/GPT4 Correct/i]],
  ['zephyr', [/<\|system\|>/i]],
  ['chatml', [/im_start|<\|im_sep\|>/i]],
  ['alpaca', [/### Instruction:/i]],
  ['llama2', [/\[INST\]/i]],
  ['vicuna', [/USER:/]],
  ['falcon', [/User: /]],
];

/** Euristiche su architettura/nome GGUF, usate se il template manca. */
const ARCH_FAMILY_HINTS = [
  [/qwen|yi|internlm|hermes|smollm|nemotron|olmo/, 'chatml'],
  [/llama.*3|llama-3/, 'llama3'],
  [/gemma/, 'gemma'],
  [/phi-?3/, 'phi3'],
  [/command-?r|cohere/, 'commandr'],
  [/chatglm|glm-?4/, 'chatglm'],
  [/deepseek/, 'deepseek'],
  [/minicpm/, 'minicpm'],
  [/granite/, 'granite'],
  [/falcon/, 'falcon'],
  [/mistral|llama.*2|vicuna/, 'llama2'],
];

/**
 * Rileva la famiglia di template dal Jinja del GGUF, con fallback su
 * architettura e nome del modello. Ritorna sempre un nome presente in
 * TEMPLATE_BUILDERS.
 */
function detectFamily(template, architecture = '', name = '') {
  const jinja = String(template || '');
  for (const [family, patterns] of TEMPLATE_MARKERS) {
    if (patterns.some((re) => re.test(jinja))) return family;
  }

  const label = `${architecture} ${name}`.toLowerCase();
  for (const [re, family] of ARCH_FAMILY_HINTS) {
    if (re.test(label)) return family;
  }
  return 'chatml';
}

const STOP_MARKER_RE = /<[\uFF5C|][^<>\n]{0,64}[\uFF5C|]>|<[A-Za-z_][A-Za-z0-9_]{0,40}>/g;

/** Estrae i possibili token di stop dai marker presenti nel template Jinja. */
function detectStopStrings(template, family) {
  const stops = new Set(FAMILY_STOP_TOKENS[family] || []);
  const jinja = String(template || '');

  for (const match of jinja.match(STOP_MARKER_RE) || []) {
    if (!/end|eot|eos|eom|eot_id/i.test(match)) continue;
    if (/start|begin|im_start|end_of_role/i.test(match)) continue;
    stops.add(match);
  }
  return [...stops].slice(0, 40);
}

/* ---------------------------------------------------------------------------
 * Preflight GGUF: legge solo l'header per ricavare architettura e n_ctx_train
 * ------------------------------------------------------------------------- */

const GGUF_MAGIC = 0x46554747;
const GGUF_PROBE_BYTES = 2 * 1024 * 1024;
const ggufHeaderCache = new Map();

const GGUF_SCALAR_SIZES = [1, 1, 2, 2, 4, 4, 4, 1, -1, -1, 8, 8, 8];
const GGUF_NEED_MORE = Symbol('gguf-need-more');

/**
 * Legge l'header GGUF (v2/v3) e restituisce architettura, nome e context
 * length di training, senza scaricare l'intero file.
 *
 * @param {Uint8Array} bytes
 * @returns {{architecture: string|null, name: string|null, contextTrain: number|null, metadata: Object}}
 */
function parseGgufHeader(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let offset = 0;

  const need = (n) => {
    if (offset + n > bytes.byteLength) throw GGUF_NEED_MORE;
  };
  const u32 = () => {
    need(4);
    const value = view.getUint32(offset, true);
    offset += 4;
    return value;
  };
  const u64 = () => {
    need(8);
    const value = view.getBigUint64(offset, true);
    offset += 8;
    return Number(value);
  };
  const str = () => {
    const len = u64();
    if (len > bytes.byteLength) throw GGUF_NEED_MORE;
    need(len);
    const value = new TextDecoder().decode(bytes.subarray(offset, offset + len));
    offset += len;
    return value;
  };
  const skip = (n) => {
    need(n);
    offset += n;
  };

  if (bytes.byteLength < 8) throw GGUF_NEED_MORE;
  if (u32() !== GGUF_MAGIC) throw new Error('Il file non è in formato GGUF');
  const version = u32();
  if (version < 2 || version > 3) throw new Error(`Versione GGUF non supportata: ${version}`);

  const tensorCount = version >= 2 ? u64() : 0;
  const kvCount = version >= 2 ? u64() : 0;
  if (kvCount > 1_000_000) throw new Error('Header GGUF non valido (troppe chiavi)');

  const metadata = {};
  let architecture = null;
  let name = null;
  let contextTrain = null;

  const readValue = (type, key) => {
    switch (type) {
      case 0:
      case 1:
      case 7:
        offset += 1;
        need(0);
        return null;
      case 2:
      case 3:
        offset += 2;
        need(0);
        return null;
      case 4:
      case 5:
      case 6:
        offset += 4;
        need(0);
        return null;
      case 10:
      case 11:
      case 12:
        offset += 8;
        need(0);
        return null;
      case 8: {
        const value = str();
        if (key === 'general.architecture') architecture = value;
        else if (key === 'general.name') name = value;
        else if (!contextTrain && /context_length$|\.n_ctx_train$/.test(key)) {
          const parsed = Number.parseInt(value, 10);
          if (Number.isFinite(parsed) && parsed > 0) contextTrain = parsed;
        }
        if (Object.keys(metadata).length < 64) metadata[key] = value;
        return value;
      }
      case 9: {
        const itemType = u32();
        const count = u64();
        if (count > 10_000_000) throw new Error('Header GGUF non valido (array enorme)');
        const itemSize = GGUF_SCALAR_SIZES[itemType];
        if (itemSize == null) throw new Error(`Tipo GGUF non supportato: ${itemType}`);
        if (itemType === 8) {
          for (let i = 0; i < count; i += 1) str();
        } else if (itemSize > 0) {
          skip(itemSize * count);
        } else {
          for (let i = 0; i < count; i += 1) readValue(itemType, key);
        }
        return null;
      }
      default:
        throw new Error(`Tipo GGUF non supportato: ${type}`);
    }
  };

  for (let i = 0; i < kvCount; i += 1) {
    const key = str();
    const type = u32();
    if (type === 4) {
      need(4);
      const value = view.getUint32(offset, true);
      offset += 4;
      if (!contextTrain && /context_length$|\.n_ctx_train$/.test(key)) contextTrain = value;
      if (Object.keys(metadata).length < 64) metadata[key] = value;
    } else if (type === 10) {
      need(8);
      const value = Number(view.getBigUint64(offset, true));
      offset += 8;
      if (!contextTrain && /context_length$|\.n_ctx_train$/.test(key)) contextTrain = value;
    } else {
      readValue(type, key);
    }

    // Abbiamo tutto ciò che serve: interrompiamo la lettura.
    if (architecture && contextTrain) break;
  }

  if (tensorCount > 1_000_000_000) throw new Error('Header GGUF non valido');
  return { architecture, name, contextTrain, metadata };
}

/**
 * Scarica solo i primi byte di un GGUF remoto (Range request) e ne legge
 * l'header. Ritorna null se non è possibile (CORS, server senza Range, ecc.).
 */
async function readGgufHeaderInfo(url, maxBytes = GGUF_PROBE_BYTES) {
  const target = toAbsoluteUrl(normalizeModelUrl(url));
  if (!/^https?:/i.test(target)) return null;
  if (ggufHeaderCache.has(target)) return ggufHeaderCache.get(target);

  const task = (async () => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 15000);
    try {
      const response = await fetch(target, {
        headers: { Range: `bytes=0-${maxBytes - 1}` },
        signal: controller.signal,
        cache: 'default',
      });
      if (!response.ok && response.status !== 206) return null;
      const reader = response.body?.getReader();
      if (!reader) return null;

      let buffer = new Uint8Array(0);
      try {
        while (buffer.byteLength < maxBytes) {
          const { done, value } = await reader.read();
          if (done) break;
          const next = new Uint8Array(buffer.byteLength + value.byteLength);
          next.set(buffer, 0);
          next.set(value, buffer.byteLength);
          buffer = next;
          try {
            return parseGgufHeader(buffer);
          } catch (err) {
            if (err !== GGUF_NEED_MORE) return null;
          }
        }
      } finally {
        try {
          await reader.cancel();
        } catch {
          /* ignora */
        }
      }
      return null;
    } catch {
      return null;
    } finally {
      clearTimeout(timer);
    }
  })();

  ggufHeaderCache.set(target, task);
  return task;
}

/* ---------------------------------------------------------------------------
 * Classe principale
 * ------------------------------------------------------------------------- */

export class GGUFEngine {
  /**
   * @param {Object} config
   * @param {string} [config.modelUrl]             URL del file .gguf (obbligatorio senza un modello GPU).
   * @param {'auto'|'wasm'|'webgpu'} [config.backend='auto'] Backend di inferenza: GPU se possibile, altrimenti GGUF/WASM.
   * @param {string} [config.webgpuModel]          Id del modello MLC per il backend GPU.
   * @param {Object} [config.webgpuModelRecord]   ModelRecord MLC custom (self-hosting).
   * @param {string} [config.webllmModuleUrl]     Modulo ESM di WebLLM (default: CDN).
   * @param {number|'auto'} [config.webgpuContextSize=4096] Contesto dichiarato al backend GPU.
   * @param {boolean}[config.webgpuFallbackToWasm=true] Con backend 'auto', ripiega su GGUF/WASM.
   * @param {string} [config.wllamaModuleUrl]     URL del modulo ESM di Wllama (default: vendor locale).
   * @param {Object} [config.wasmPaths]           Mappa dei path dei binari WASM (v3: { default }). (default: vendor locale).
   * @param {{worker:string, wasm:string}} [config.wllamaCompatPaths] Worker/WASM compat per Safari (default: vendor locale).
   * @param {boolean}[config.autoFallback=true]   Se i file in "vendor" mancano, usa i CDN web.
   * @param {string} [config.wllamaCdnVersion]    Versione di Wllama da usare come ripiego web.
   * @param {string} [config.modelFallbackUrl]    URL GGUF alternativo se il modello locale manca.
   * @param {string[]} [config.modelFallbackUrls] Più URL GGUF alternativi, in ordine di tentativo.
   * @param {boolean}[config.allowOffline]        Consenti l'uso della cache offline.
   * @param {string} [config.systemPrompt]        Prompt di sistema predefinito (default: nessuno).
   * @param {number|'auto'|'max'} [config.contextSize='auto']  n_ctx: 'auto' = min(n_ctx_train, 4096), 'max' = n_ctx_train (max 32768).
   * @param {number|'auto'} [config.batchSize='auto']        n_batch (default: min(256, n_ctx)).
   * @param {number} [config.threads]             Thread WASM (default: metà dei core).
   * @param {number} [config.maxTokens=256]       Token massimi per risposta.
   * @param {number} [config.temperature=0.7]     Temperatura di campionamento.
   * @param {number} [config.topK=40]
   * @param {number} [config.topP=0.9]
   * @param {number} [config.minP=0.05]
   * @param {number} [config.repeatPenalty=1.1]
   * @param {number} [config.seed=-1]
   * @param {string|Function} [config.chatTemplate='auto']  'auto' = template Jinja del GGUF, con fallback automatico.
   * @param {boolean}[config.trimHistory=true]    Adatta la cronologia al contesto.
   * @param {boolean}[config.autoDetect=true]     Rileva template e token di stop dai metadati GGUF.
   * @param {boolean}[config.preflightMetadata=true] Legge l'header GGUF remoto per n_ctx_train.
   */
  constructor(config = {}) {
    const hasModelUrl =
      typeof config.modelUrl === 'string' && config.modelUrl.trim() !== '';
    const hasGpuModel = Boolean(
      (typeof config.webgpuModel === 'string' && config.webgpuModel.trim()) ||
        config.webgpuModelRecord?.model_id
    );

    if (!hasModelUrl && !hasGpuModel) {
      throw new TypeError(
        'GGUFEngine: serve "modelUrl" (GGUF) oppure "webgpuModel"/"webgpuModelRecord" (backend GPU).'
      );
    }

    // Sempre assoluto: il download avviene in una Web Worker da blob URL
    this.modelUrl = hasModelUrl
      ? toAbsoluteUrl(normalizeModelUrl(config.modelUrl))
      : null;
    this.wllamaModuleUrl = config.wllamaModuleUrl || DEFAULT_WLLAMA_MODULE;
    this.wasmPaths = normalizeWasmPaths(config.wasmPaths || DEFAULT_WASM_PATHS);
    this.wllamaCompatPaths = config.wllamaCompatPaths || {
      ...DEFAULT_WLLAMA_COMPAT_PATHS,
    };
    this.allowOffline = config.allowOffline === true;

    // Ripiego automatico su fonti web quando gli asset locali non ci sono.
    this._autoFallback = config.autoFallback !== false;
    this._wllamaCdnVersion = config.wllamaCdnVersion || null;
    this._customModuleUrl = config.wllamaModuleUrl || null;
    this._customWasmPaths = config.wasmPaths ? normalizeWasmPaths(config.wasmPaths) : null;
    this._customCompatPaths = config.wllamaCompatPaths || null;
    this.runtimeSource = null;

    const fallbacks = Array.isArray(config.modelFallbackUrls)
      ? config.modelFallbackUrls
      : config.modelFallbackUrl
        ? [config.modelFallbackUrl]
        : [];
    this.modelFallbacks = fallbacks
      .filter((url) => typeof url === 'string' && url.trim())
      .map((url) => toAbsoluteUrl(normalizeModelUrl(url.trim())));
    this._activeModelUrl = this.modelUrl;

    this.options = {
      ...DEFAULT_OPTIONS,
      ...Object.fromEntries(
        Object.entries(config).filter(([, value]) => value !== undefined)
      ),
    };

    // Cronologia per la modalità stateful (send/resetHistory)
    /** @type {{role: string, content: string}[]} */
    this.messages = [];
    this.systemPrompt = this.options.systemPrompt;

    // Stato interno
    this._wllama = null;
    this._gpu = null;
    this._gpuInfo = null;
    this._backend = null;
    this._ready = false;
    this._disposed = false;
    this._loadPromise = null;
    this._busy = false;
    this._abortRequested = false;
    this._status = 'idle';
    this._lastResult = null;
    this._listeners = new Map();
    this._lock = Promise.resolve();

    // Rilevamento automatico (template, contesto, token di stop)
    this._preflight = null;
    this._detected = null;
    this._chatTemplateRaw = null;
    this._nativeTemplateOk = null;
    this._autoContext = null;
    this._abortController = null;
    this._libllamaVersion = null;
  }

  /* ----------------------------- Eventi ----------------------------- */

  /**
   * Registra un listener.
   * Eventi: 'status', 'progress', 'token', 'done', 'abort', 'error'.
   * @returns {() => void} funzione per rimuovere il listener.
   */
  on(event, handler) {
    if (typeof handler !== 'function') {
      throw new TypeError('GGUFEngine.on: handler deve essere una funzione.');
    }
    if (!this._listeners.has(event)) this._listeners.set(event, new Set());
    this._listeners.get(event).add(handler);
    return () => this.off(event, handler);
  }

  off(event, handler) {
    this._listeners.get(event)?.delete(handler);
  }

  #emit(event, payload) {
    for (const handler of this._listeners.get(event) || []) {
      try {
        handler(payload);
      } catch (err) {
        console.warn(`[GGUFEngine] listener "${event}" ha generato un errore:`, err);
      }
    }
  }

  #setStatus(status) {
    this._status = status;
    this.#emit('status', status);
  }

  /* --------------------------- Stato pubblico --------------------------- */

  get status() {
    return this._status;
  }

  get isLoaded() {
    return this._ready;
  }

  get isBusy() {
    return this._busy;
  }

  /** Istanza Wllama sottostante (per uso avanzato), o null se non caricata. */
  get raw() {
    return this._wllama;
  }

  /** Istanza WebLLM sottostante (backend GPU), o null se non attivo. */
  get rawGpu() {
    return this._gpu?.engine ?? null;
  }

  /** Backend in uso: 'wasm', 'webgpu' oppure null prima di load(). */
  get backend() {
    return this._backend;
  }

  get isWebGPU() {
    return this._backend === 'webgpu';
  }

  /** Info sull'adattatore WebGPU (null se non rilevato o non disponibile). */
  get gpuInfo() {
    return this._gpuInfo ? { ...this._gpuInfo } : null;
  }

  /** Etichetta leggibile del backend attivo (es. "WebGPU · NVIDIA ..."). */
  get backendLabel() {
    if (this._backend === 'webgpu') {
      return describeAdapter(this._gpuInfo?.adapterInfo);
    }
    if (this._backend === 'wasm') return 'CPU (WASM)';
    return 'non caricato';
  }

  get lastResult() {
    return this._lastResult ? { ...this._lastResult } : null;
  }

  /** URL del modello effettivamente usato (locale o ripiego web). */
  get activeModelUrl() {
    return this._activeModelUrl;
  }

  /**
   * Versione di llama.cpp inclusa nel runtime Wllama (es. "b6327-4d74393").
   * null prima del load o con Wllama < 2.3.3, che non espone l'informazione.
   */
  get libllamaVersion() {
    return this._libllamaVersion;
  }

  /** Da dove arriva il runtime: 'local', 'cdn' o 'custom' (null prima del load). */
  get runtimeSource() {
    return this._runtimeSource;
  }

  set runtimeSource(value) {
    this._runtimeSource = value;
  }

  /** Informazioni sul modello, disponibili dopo load(). */
  getModelInfo() {
    if (this._backend === 'webgpu') {
      return {
        backend: 'webgpu',
        model: this._activeModelUrl,
        gpu: this._gpuInfo,
        detected: this.detected,
      };
    }
    if (!this._wllama) return this._preflight ? { preflight: this._preflight, detected: null } : null;
    try {
      const meta = this._wllama.getModelMetadata?.() ?? null;
      return meta ? { ...meta, backend: 'wasm', detected: this.detected } : null;
    } catch {
      return null;
    }
  }

  /** Configurazione rilevata automaticamente dal modello (null prima del load). */
  get detected() {
    if (!this._detected) return null;
    return { ...this._detected, stopStrings: [...this._detected.stopStrings] };
  }

  /** Template Jinja contenuto nel GGUF (null se assente). */
  get chatTemplateRaw() {
    return this._chatTemplateRaw;
  }

  /* ----------------------------- Caricamento ----------------------------- */

  #threadsToUse() {
    const hw = hardwareThreads();
    const wanted =
      this.options.threads == null
        ? Math.max(1, Math.floor(hw / 2))
        : clampInt(this.options.threads, 1, hw);
    return multiThreadAvailable() ? wanted : 1;
  }

  /**
   * Scarica (o recupera dalla cache) e inizializza il modello.
   * È idempotente: chiamate concorrenti condividono la stessa Promise.
   *
   * @param {{ onProgress?: (p: {loaded: number, total: number, percent: number}) => void }} [opts]
   * @returns {Promise<GGUFEngine>}
   */
  async load({ onProgress } = {}) {
    if (this._disposed) {
      throw new Error('GGUFEngine: istanza già rilasciata (dispose).');
    }
    if (this._ready) return this;

    if (onProgress) {
      const remove = this.on('progress', onProgress);
      return this.#startLoad().finally(remove);
    }
    return this.#startLoad();
  }

  /**
   * Determina modulo runtime e path WASM: locali se presenti, altrimenti CDN.
   * @returns {Promise<{moduleUrl:string, wasmPaths:Object, compat:Object|null, source:string}>}
   */
  async #resolveRuntimeSources() {
    const localSources = {
      moduleUrl: toAbsoluteUrl(DEFAULT_WLLAMA_MODULE, import.meta.url),
      wasmPaths: Object.fromEntries(
        Object.entries(DEFAULT_WASM_PATHS).map(([key, value]) => [
          key,
          toAbsoluteUrl(value, import.meta.url),
        ])
      ),
      compat: Object.fromEntries(
        Object.entries(DEFAULT_WLLAMA_COMPAT_PATHS).map(([key, value]) => [
          key,
          toAbsoluteUrl(value, import.meta.url),
        ])
      ),
      source: 'local',
    };

    // Path espliciti forniti dall'utente: li rispettiamo, ma l'import del
    // modulo ha comunque un ultimo ripiego sul CDN in #importRuntime().
    if (this._customModuleUrl || this._customWasmPaths || this._customCompatPaths) {
      const base = this._autoFallback
        ? await resolveDefaultRuntimeSources(this._wllamaCdnVersion)
        : localSources;
      return {
        moduleUrl: this._customModuleUrl
          ? toAbsoluteUrl(this._customModuleUrl, import.meta.url)
          : base.moduleUrl,
        wasmPaths: this._customWasmPaths
          ? normalizeWasmPaths(this._customWasmPaths)
          : { ...base.wasmPaths },
        compat: this._customCompatPaths
          ? { ...this._customCompatPaths }
          : base.compat
            ? { ...base.compat }
            : null,
        source: 'custom',
      };
    }

    const resolved = this._autoFallback
      ? await resolveDefaultRuntimeSources(this._wllamaCdnVersion)
      : localSources;

    return {
      ...resolved,
      wasmPaths: { ...resolved.wasmPaths },
      compat: resolved.compat ? { ...resolved.compat } : null,
    };
  }

  /**
   * Importa Wllama e, se il modulo locale fallisce, riprova con il CDN.
   * @returns {Promise<{Wllama: Function, wasmPaths: Object, compat: Object|null}>}
   */
  async #importRuntime(sources) {
    const load = async (candidate) => {
      const mod = await import(/* @vite-ignore */ candidate.moduleUrl);
      const Wllama = mod.Wllama || mod.default?.Wllama || mod.default;
      if (typeof Wllama !== 'function') {
        throw new Error(
          `il modulo "${candidate.moduleUrl}" non esporta la classe "Wllama".`
        );
      }
      return Wllama;
    };

    const failure = (err, moduleUrl) => {
      const wrapped = friendlyLoadError(err);
      wrapped.message = `Impossibile caricare il runtime Wllama da "${moduleUrl}": ${wrapped.message}`;
      return wrapped;
    };

    try {
      const Wllama = await load(sources);
      this.runtimeSource = sources.source;
      return {
        Wllama,
        wasmPaths: { ...sources.wasmPaths },
        compat: sources.compat ? { ...sources.compat } : null,
      };
    } catch (err) {
      if (!this._autoFallback || sources.source === 'cdn') {
        throw failure(err, sources.moduleUrl);
      }

      const cdn = { ...cdnRuntimeSources(this._wllamaCdnVersion), source: 'cdn' };
      try {
        const Wllama = await load(cdn);
        this.runtimeSource = 'cdn';
        this.#emit('fallback', {
          asset: 'runtime',
          from: sources.moduleUrl,
          to: cdn.moduleUrl,
        });
        return {
          Wllama,
          wasmPaths: { ...cdn.wasmPaths },
          compat: cdn.compat ? { ...cdn.compat } : null,
        };
      } catch (cdnErr) {
        throw failure(cdnErr, cdn.moduleUrl);
      }
    }
  }

  /** Modelli da provare: il locale e, se configurati, i ripieghi web. */
  #modelCandidates() {
    const candidates = [];
    if (this.modelUrl) candidates.push({ url: this.modelUrl, source: 'local' });
    if (!this._autoFallback) return candidates;

    for (const fallback of this.modelFallbacks) {
      if (!candidates.some((candidate) => candidate.url === fallback)) {
        candidates.push({ url: fallback, source: 'remote' });
      }
    }
    return candidates;
  }

  /** Evita di ritentare un modello remoto dopo errori non di rete (es. OOM). */
  #canFallbackModel(err) {
    const raw = err?.message || String(err || '');
    if (/abort/i.test(raw)) return false;
    if (isOutOfMemoryError(err)) return false;
    return true;
  }

  /** Dimensionamento automatico attivo? */
  #autoSizing() {
    return this.options.autoDetect !== false;
  }

  /**
   * Contesto effettivo (n_ctx).
   * - numero: usato così com'è;
   * - 'auto': min(n_ctx_train, 4096) oppure 4096;
   * - 'max': min(n_ctx_train, 32768).
   */
  #contextSize() {
    const requested = this.options.contextSize;
    if (typeof requested === 'number' && Number.isFinite(requested) && requested > 0) {
      return Math.floor(requested);
    }

    const train = Number(this._detected?.contextTrain || this._preflight?.contextTrain) || 0;
    const fallback = train > 0 ? train : 4096;
    let size = requested === 'max' ? Math.min(fallback, 32768) : Math.min(fallback, 4096);

    // Dimensionamento ridotto automaticamente dopo un errore di memoria.
    if (this._autoContext) size = Math.min(size, this._autoContext);
    if (train > 0) size = Math.min(size, train);

    return Math.max(256, Math.floor(size / 256) * 256);
  }

  /** Batch effettivo (n_batch), sempre <= n_ctx. */
  #batchSize() {
    const context = this.#contextSize();
    const configured = this.options.batchSize;
    if (typeof configured === 'number' && Number.isFinite(configured) && configured > 0) {
      return Math.min(Math.floor(configured), context);
    }
    return Math.max(1, Math.min(256, context));
  }

  /** Legge (con cache) architettura e n_ctx_train dall'header GGUF remoto. */
  async #preflightFor(url) {
    if (!this.#autoSizing() || this.options.preflightMetadata === false) return null;
    const info = await readGgufHeaderInfo(url);
    if (info) this._preflight = info;
    return info;
  }

  /**
   * Rileva dal modello caricato: famiglia di template, contesto di training,
   * presenza del template Jinja e token di stop.
   */
  async #detectModel(wllama, preflight) {
    const meta = (() => {
      try {
        return wllama.getModelMetadata?.() ?? null;
      } catch {
        return null;
      }
    })();
    const contextInfo = (() => {
      try {
        return wllama.getLoadedContextInfo?.() ?? null;
      } catch {
        return null;
      }
    })();

    const architecture =
      preflight?.architecture || meta?.meta?.['general.architecture'] || '';
    const name = preflight?.name || meta?.meta?.['general.name'] || '';
    const trainCandidates = [
      preflight?.contextTrain,
      meta?.hparams?.nCtxTrain,
      contextInfo?.n_ctx_train,
    ].map(Number);
    const contextTrain = trainCandidates.find((value) => Number.isFinite(value) && value > 0) || null;

    let template = null;
    try {
      template = wllama.getChatTemplate?.() ?? null;
    } catch {
      template = null;
    }
    this._chatTemplateRaw = typeof template === 'string' && template.trim() ? template : null;

    const family = detectFamily(this._chatTemplateRaw, architecture, name);
    const stopStrings = detectStopStrings(this._chatTemplateRaw, family);
    const hasChatTemplate = this._chatTemplateRaw != null;

    // Wllama 3.x non espone più detokenize/tokenize: i token EOG sono già
    // gestiti dal runtime e dall'API chat, quindi qui restano i marker del
    // template (usati per sanitizzare l'output e come `stop` testuali).

    this._detected = {
      family,
      architecture,
      name,
      contextTrain,
      contextSize: 0,
      hasChatTemplate,
      stopStrings,
      templateSource: hasChatTemplate ? 'native' : 'builder',
    };
    this._detected.contextSize = this.#contextSize();
  }

  /** Crea un'istanza Wllama e carica il modello con le dimensioni correnti. */
  async #tryLoad(Wllama, wasmPaths, url, nThreads, compat) {
    const instance = new Wllama(wasmPaths, {
      parallelDownloads: this.options.parallelDownloads,
      ...(this.allowOffline ? { allowOffline: true } : {}),
    });

    // Se il pacchetto compat è self-hosted, evita il fetch automatico dal CDN
    // sui browser senza JSPI/MEMORY64.
    if (compat && typeof instance.setCompat === 'function') {
      instance.setCompat(compat, 'safari');
    }

    await instance.loadModelFromUrl(url, {
      n_ctx: this.#contextSize(),
      n_batch: this.#batchSize(),
      n_threads: nThreads,
      // Backend CPU: WebGPU resta opt-in (Wllama lo attiva con n_gpu_layers>0).
      n_gpu_layers: 0,
      progressCallback: ({ loaded, total }) => {
        const percent = total > 0 ? (loaded / total) * 100 : 0;
        this.#emit('progress', { loaded, total, percent });
      },
    }).catch(async (err) => {
      try {
        await instance.exit();
      } catch {
        /* ignora */
      }
      throw err;
    });

    return instance;
  }

  /** Contesto dichiarato al backend GPU (modelRecord oppure webgpuContextSize). */
  #webgpuContextSize() {
    const record = this.options.webgpuModelRecord;
    const fromRecord = Number(record?.overrides?.context_window_size);
    if (Number.isFinite(fromRecord) && fromRecord > 0) {
      return Math.floor(fromRecord);
    }
    const configured = Number(this.options.webgpuContextSize);
    if (Number.isFinite(configured) && configured > 0) {
      return Math.min(131072, Math.max(512, Math.floor(configured)));
    }
    return 4096;
  }

  /** Id del modello GPU configurato, o null. */
  #webgpuModelId() {
    const record = this.options.webgpuModelRecord;
    const id = this.options.webgpuModel || record?.model_id || null;
    return typeof id === 'string' && id.trim() ? id.trim() : null;
  }

  /**
   * URL del modulo WebLLM: esplicito, poi vendor locale, poi CDN (default del
   * backend quando la funzione ritorna undefined).
   */
  async #resolveWebLLMModuleUrl() {
    const configured = this.options.webllmModuleUrl;
    if (configured) return toAbsoluteUrl(configured);
    if (await assetExists(DEFAULT_WEBLLM_VENDOR_MODULE)) {
      return toAbsoluteUrl(DEFAULT_WEBLLM_VENDOR_MODULE);
    }
    return undefined;
  }

  /**
   * Prova ad attivare il backend WebGPU.
   * @param {'auto'|'webgpu'} requested
   * @returns {Promise<boolean>} true se il backend GPU è pronto.
   */
  async #startWebGPU(requested) {
    const model = this.#webgpuModelId();
    if (!model) {
      if (requested === 'webgpu') {
        throw gpuError(
          new Error(
            'Nessun modello GPU configurato: imposta webgpuModel (id MLC) oppure webgpuModelRecord.'
          )
        );
      }
      return false;
    }

    const detection = await WebGPUBackend.detect();
    this._gpuInfo = detection;

    if (!detection.supported) {
      const err = gpuError(
        new Error(detection.reason || 'WebGPU non disponibile.')
      );
      if (
        requested === 'webgpu' ||
        this.options.webgpuFallbackToWasm === false
      ) {
        throw err;
      }
      this.#emit('fallback', {
        asset: 'backend',
        from: 'webgpu',
        to: 'wasm',
        reason: detection.reason,
      });
      return false;
    }

    const backend = new WebGPUBackend({
      model,
      modelRecord: this.options.webgpuModelRecord || null,
      moduleUrl: await this.#resolveWebLLMModuleUrl(),
      contextSize: this.#webgpuContextSize(),
    });

    try {
      await backend.load({
        onProgress: ({ progress, text }) => {
          this.#emit('progress', {
            loaded: progress,
            total: 1,
            percent: Math.max(0, Math.min(100, progress * 100)),
            text,
          });
        },
      });
    } catch (err) {
      const wrapped = err?.name === 'WebGPULoadError' ? err : gpuError(err);
      if (
        requested === 'webgpu' ||
        this.options.webgpuFallbackToWasm === false
      ) {
        throw wrapped;
      }
      this.#emit('fallback', {
        asset: 'backend',
        from: 'webgpu',
        to: 'wasm',
        reason: String(err?.message || err),
      });
      return false;
    }

    this._gpu = backend;
    this._backend = 'webgpu';
    this._gpuInfo = backend.info || detection;
    this._activeModelUrl = model;
    this._detected = {
      family: 'webllm',
      architecture: 'mlc',
      name: model,
      contextTrain: this.#webgpuContextSize(),
      contextSize: this.#webgpuContextSize(),
      hasChatTemplate: true,
      stopStrings: [],
      templateSource: 'webllm',
    };
    this.#emit('backend', {
      backend: 'webgpu',
      model,
      info: this._gpuInfo,
    });
    return true;
  }

  #startLoad() {
    if (this._loadPromise) return this._loadPromise;

    this._loadPromise = (async () => {
      this.#setStatus('loading');
      this._backend = null;

      const requested =
        this.options.backend === 'wasm'
          ? 'wasm'
          : this.options.backend === 'webgpu'
            ? 'webgpu'
            : 'auto';

      if (requested !== 'wasm') {
        const gpuReady = await this.#startWebGPU(requested);
        if (gpuReady) {
          this._ready = true;
          this._lastResult = null;
          this.#setStatus('ready');
          return this;
        }
      }

      return this.#loadWasm();
    })().catch((err) => {
      this._loadPromise = null;
      this._wllama = null;
      this._gpu = null;
      this._backend = null;
      this._ready = false;
      this._detected = null;
      this.#setStatus('error');
      const wrapped =
        err?.name === 'GGUFLoadError' || err?.name === 'WebGPULoadError'
          ? err
          : friendlyLoadError(err);
      this.#emit('error', wrapped);
      throw wrapped;
    });

    return this._loadPromise;
  }

  /** Percorso Wllama (WASM/CPU) con un modello GGUF. */
  async #loadWasm() {
    if (!this.modelUrl) {
      throw new Error(
        'GGUFEngine: il backend GPU non è disponibile e non è stato configurato un modelUrl GGUF. ' +
          'Indica un modello GGUF oppure usa backend: "webgpu" con un modello compatibile.'
      );
    }

    const sources = await this.#resolveRuntimeSources();
    const { Wllama, wasmPaths, compat } = await this.#importRuntime(sources);

      try {
        this._libllamaVersion = Wllama.getLibllamaVersion?.() ?? null;
      } catch {
        this._libllamaVersion = null;
      }

      const nThreads = this.#threadsToUse();
      const candidates = this.#modelCandidates();
      let lastError = null;
      let lastArchitecture = null;

      for (let index = 0; index < candidates.length; index++) {
        const candidate = candidates[index];

        const preflight = await this.#preflightFor(candidate.url);
        if (preflight?.architecture) lastArchitecture = preflight.architecture;
        this._autoContext = null;
        this._detected = null;
        this._nativeTemplateOk = null;

        try {
          let instance = null;
          let attempts = 0;

          // Se il contesto automatico non entra in memoria, dimezzalo e ritenta.
          for (;;) {
            try {
              instance = await this.#tryLoad(
                Wllama,
                wasmPaths,
                candidate.url,
                nThreads,
                compat
              );
              break;
            } catch (err) {
              const previous = this.#contextSize();
              const canRetry =
                this.#autoSizing() &&
                typeof this.options.contextSize !== 'number' &&
                this.options.contextSize !== 'max' &&
                isOutOfMemoryError(err) &&
                attempts < 4 &&
                previous > 512;

              if (!canRetry) throw err;

              this._autoContext = Math.max(512, Math.floor(previous / 2));
              attempts += 1;
              this.#emit('fallback', {
                asset: 'context',
                from: previous,
                to: this.#contextSize(),
                reason: String(err?.message || err),
              });
            }
          }

          await this.#detectModel(instance, preflight);

          this._wllama = instance;
          this._backend = 'wasm';
          this._activeModelUrl = candidate.url;
          this._ready = true;
          this._lastResult = null;
          this.#emit('backend', { backend: 'wasm' });
          this.#setStatus('ready');
          return this;
        } catch (err) {
          lastError = err;
          this._wllama = null;

          const hasNext = index < candidates.length - 1;
          if (!hasNext || !this.#canFallbackModel(err)) break;

          this.#emit('fallback', {
            asset: 'model',
            from: candidate.url,
            to: candidates[index + 1].url,
          });
        }
      }

      throw friendlyLoadError(lastError, {
        architecture: lastArchitecture,
        libVersion: this._libllamaVersion,
      });
  }

  /**
   * Modifica le opzioni di default (non richiede ricaricare il modello,
   * a meno di contextSize/threads/batchSize/backend che vengono fissati al load).
   */
  configure(partial = {}) {
    const reloadKeys = [
      'contextSize',
      'batchSize',
      'threads',
      'backend',
      'webgpuModel',
      'webgpuModelRecord',
      'webllmModuleUrl',
      'webgpuContextSize',
      'webgpuFallbackToWasm',
      'wllamaModuleUrl',
      'wasmPaths',
      'wllamaCompatPaths',
      'wllamaCdnVersion',
      'autoFallback',
    ];
    const needsReload = reloadKeys.some(
      (key) => key in partial && partial[key] !== this.options[key]
    );

    this.options = { ...this.options, ...partial };
    if ('systemPrompt' in partial) this.systemPrompt = partial.systemPrompt;
    if ('contextSize' in partial) this._autoContext = null;
    if ('chatTemplate' in partial) this._nativeTemplateOk = null;
    if ('wllamaModuleUrl' in partial) {
      this.wllamaModuleUrl = partial.wllamaModuleUrl || DEFAULT_WLLAMA_MODULE;
      this._customModuleUrl = partial.wllamaModuleUrl || null;
    }
    if ('wasmPaths' in partial) {
      this.wasmPaths = normalizeWasmPaths(partial.wasmPaths || DEFAULT_WASM_PATHS);
      this._customWasmPaths = partial.wasmPaths
        ? normalizeWasmPaths(partial.wasmPaths)
        : null;
    }
    if ('wllamaCompatPaths' in partial) {
      this.wllamaCompatPaths = partial.wllamaCompatPaths || {
        ...DEFAULT_WLLAMA_COMPAT_PATHS,
      };
      this._customCompatPaths = partial.wllamaCompatPaths || null;
    }
    if ('wllamaCdnVersion' in partial) {
      this._wllamaCdnVersion = partial.wllamaCdnVersion || null;
    }
    if ('autoFallback' in partial) this._autoFallback = partial.autoFallback !== false;

    return { needsReload, options: { ...this.options } };
  }

  /* ----------------------------- Mutex ----------------------------- */

  #acquire() {
    let release;
    const previous = this._lock;
    this._lock = new Promise((resolve) => {
      release = resolve;
    });
    return previous.then(() => release);
  }

  /* --------------------------- Prompt / template --------------------------- */

  #normalizeMessages(input) {
    if (typeof input === 'string') {
      return [{ role: 'user', content: input }];
    }
    if (!Array.isArray(input) || input.length === 0) {
      throw new TypeError(
        'GGUFEngine: "messages" deve essere un array non vuoto di { role, content }.'
      );
    }

    return input.map((msg) => {
      if (typeof msg === 'string') return { role: 'user', content: msg };
      if (!msg || typeof msg !== 'object') {
        throw new TypeError('GGUFEngine: messaggio non valido.');
      }
      const role =
        typeof msg.role === 'string' && msg.role.trim()
          ? msg.role.trim().toLowerCase()
          : 'user';
      return { role, content: String(msg.content ?? '') };
    });
  }

  /** Famiglia di template usata per i messaggi (per i builder JS). */
  #messagesTemplateFamily() {
    const template = this.options.chatTemplate;
    if (typeof template === 'function') return 'custom';
    if (typeof template === 'string' && template !== 'auto') {
      return TEMPLATE_BUILDERS[template] ? template : 'chatml';
    }
    return this._detected?.family || 'chatml';
  }

  #withSystem(messages, systemPrompt) {
    const hasSystem = messages.some((m) => m.role === 'system');
    const system =
      systemPrompt === null || systemPrompt === undefined
        ? ''
        : String(systemPrompt).trim();

    if (hasSystem || !system) return messages;
    // Il template "plain" non ha un ruolo system: il testo verrebbe ignorato.
    if (this.#messagesTemplateFamily() === 'plain') return messages;
    return [{ role: 'system', content: system }, ...messages];
  }

  /** Stima grezza (4 caratteri/token) usata come primo filtro. */
  #trimForContext(messages, contextSize, reserve) {
    if (this.options.trimHistory === false) return messages;

    const budget = contextSize - reserve - 32;
    if (budget <= 0) return messages;

    const result = [...messages];
    const total = () => estimateTokens(result.map((m) => m.content).join('\n'));

    while (total() > budget && result.filter((m) => m.role !== 'system').length > 1) {
      const idx = result.findIndex((m) => m.role !== 'system');
      if (idx === -1) break;
      result.splice(idx, 1);
    }
    return result;
  }

  /**
   * Numero di token del prompt. Wllama 3.x non espone più tokenize: per il
   * runtime v3 si usa una stima, mentre con un eventuale runtime 2.x resta
   * disponibile il conteggio esatto.
   */
  async #countPromptTokens(prompt) {
    try {
      if (typeof this._wllama?.tokenize === 'function') {
        const tokens = await this._wllama.tokenize(prompt, true);
        if (Array.isArray(tokens)) return tokens.length + 1; // +1 per il BOS
      }
    } catch {
      /* usa la stima */
    }
    return Math.max(1, Math.ceil(prompt.length / 4));
  }

  /** L'API chat nativa di Wllama 3.x gestisce da sola il template Jinja. */
  #useNativeChat(template) {
    if (this._nativeTemplateOk === false) return false;
    if (typeof template === 'function') return false;
    if (typeof template === 'string' && template !== 'auto') return false;
    return this._detected?.hasChatTemplate === true;
  }

  /**
   * Formatta i messaggi: template Jinja nativo del GGUF quando possibile,
   * altrimenti builder JS della famiglia rilevata.
   * @returns {Promise<{prompt: string, extraStops: string[]}>}
   */
  async #formatPrompt(messages, template, opts) {
    if (typeof template === 'function') {
      const built = await template(messages, opts);
      if (built && typeof built === 'object') {
        return {
          prompt: String(built.prompt ?? ''),
          extraStops: Array.isArray(built.stopTokens) ? built.stopTokens : [],
        };
      }
      return { prompt: String(built ?? ''), extraStops: [] };
    }

    const explicit =
      typeof template === 'string' && template !== 'auto' && TEMPLATE_BUILDERS[template]
        ? template
        : null;

    const canUseNative =
      !explicit &&
      this.#autoSizing() &&
      this._nativeTemplateOk !== false &&
      typeof this._wllama?.formatChat === 'function';

    if (canUseNative) {
      try {
        const formatted = await this._wllama.formatChat(messages, true);
        if (typeof formatted === 'string' && formatted.trim()) {
          this._nativeTemplateOk = true;
          return { prompt: formatted, extraStops: [] };
        }
      } catch {
        // Template custom non gestito da wllama: si passa ai builder JS.
        this._nativeTemplateOk = false;
      }
    }

    const family = explicit || this._detected?.family || 'chatml';
    const builder = TEMPLATE_BUILDERS[family] || TEMPLATE_BUILDERS.chatml;
    return { prompt: builder(messages), extraStops: [] };
  }

  /** Token di stop (stringhe) per questa generazione. */
  #stopStringsFor(opts, extraStops = []) {
    if (Array.isArray(opts.stopTokens)) {
      return [...new Set([...opts.stopTokens, ...extraStops])].filter(Boolean);
    }
    if (Array.isArray(this.options.stopTokens)) {
      return [...new Set([...this.options.stopTokens, ...extraStops])].filter(Boolean);
    }
    const detected = this._detected?.stopStrings || [];
    return [...new Set([...detected, ...extraStops])].filter(Boolean).slice(0, 64);
  }

  /** Prompt testuale usato solo per la stima del contesto in modalità chat. */
  #chatPromptEstimate(messages) {
    return messages.map((m) => `${m.role}: ${m.content}`).join('\n');
  }

  /**
   * Prepara prompt, limite di token e stop per una generazione.
   * In modalità chat nativa il prompt non viene costruito a mano: si passano
   * i messaggi a Wllama 3.x, che applica il template Jinja del GGUF.
   * @returns {Promise<{mode:'chat'|'raw', messages?:Object[], prompt: string, promptTokens: number, nPredict: number, stopStrings: string[], templateName: string}>}
   */
  async #buildPrompt(messages, opts) {
    const template = opts.chatTemplate ?? this.options.chatTemplate;
    const withSystem = this.#withSystem(
      messages,
      opts.systemPrompt !== undefined ? opts.systemPrompt : this.systemPrompt
    );

    const contextSize = this.#contextSize();
    const requested = Number(opts.maxTokens ?? this.options.maxTokens);
    const maxTokens =
      Number.isFinite(requested) && requested > 0
        ? Math.floor(requested)
        : Math.min(256, Math.max(32, contextSize - 8));

    let trimmed = this.#trimForContext(withSystem, contextSize, maxTokens);

    // v3: chat nativa, niente prompt costruito a mano né tokenizzatore.
    if (this.#useNativeChat(template)) {
      let prompt = this.#chatPromptEstimate(trimmed);
      let promptTokens = Math.ceil(prompt.length / 4);
      let guard = 0;

      while (
        this.options.trimHistory !== false &&
        promptTokens + maxTokens + 8 > contextSize &&
        guard < 64
      ) {
        const removable = trimmed.filter((m) => m.role !== 'system').length;
        if (removable <= 1) break;
        const firstIndex = trimmed.findIndex((m) => m.role !== 'system');
        trimmed = trimmed.filter((_, i) => i !== firstIndex);
        prompt = this.#chatPromptEstimate(trimmed);
        promptTokens = Math.ceil(prompt.length / 4);
        guard += 1;
      }

      // Ultima spiaggia: tronca l'inizio dell'ultimo messaggio.
      if (promptTokens + maxTokens + 8 > contextSize && trimmed.length > 0) {
        const last = trimmed[trimmed.length - 1];
        const content = String(last.content ?? '');
        const overflowChars = Math.max(
          0,
          (promptTokens + maxTokens + 8 - contextSize) * 4
        );
        if (content.length > overflowChars + 16) {
          trimmed = [
            ...trimmed.slice(0, -1),
            { ...last, content: content.slice(overflowChars) },
          ];
          prompt = this.#chatPromptEstimate(trimmed);
          promptTokens = Math.ceil(prompt.length / 4);
        }
      }

      const stopStrings = this.#stopStringsFor(opts);
      const available = Math.max(1, contextSize - promptTokens - 2);
      return {
        mode: 'chat',
        messages: trimmed,
        prompt,
        promptTokens,
        nPredict: Math.min(maxTokens, available),
        stopStrings,
        templateName: 'native',
      };
    }

    // Builder JS / template esplicito: serve un prompt testuale.
    let formatted = await this.#formatPrompt(trimmed, template, opts);
    let promptTokens = await this.#countPromptTokens(formatted.prompt);

    // Adatta la cronologia al contesto reale del modello (tokenizzatore alla mano).
    let guard = 0;
    while (
      this.options.trimHistory !== false &&
      promptTokens + 4 > contextSize &&
      guard < 64
    ) {
      const removable = trimmed.filter((m) => m.role !== 'system').length;
      if (removable <= 1) break;
      const firstIndex = trimmed.findIndex((m) => m.role !== 'system');
      trimmed = trimmed.filter((_, i) => i !== firstIndex);
      formatted = await this.#formatPrompt(trimmed, template, opts);
      promptTokens = await this.#countPromptTokens(formatted.prompt);
      guard += 1;
    }

    // Ultima spiaggia: tronca l'inizio del contenuto dell'ultimo messaggio.
    if (promptTokens + 4 > contextSize && trimmed.length > 0) {
      const last = trimmed[trimmed.length - 1];
      const content = String(last.content ?? '');
      const overflowChars = Math.max(0, (promptTokens - contextSize + 8) * 4);
      if (content.length > overflowChars + 16) {
        trimmed = [
          ...trimmed.slice(0, -1),
          { ...last, content: content.slice(overflowChars) },
        ];
        formatted = await this.#formatPrompt(trimmed, template, opts);
        promptTokens = await this.#countPromptTokens(formatted.prompt);
      }
    }

    const stopStrings = this.#stopStringsFor(opts, formatted.extraStops);
    const available = Math.max(1, contextSize - promptTokens - 2);

    return {
      mode: 'raw',
      prompt: formatted.prompt,
      promptTokens,
      nPredict: Math.min(maxTokens, available),
      stopStrings,
      templateName: typeof template === 'string' ? template : 'custom',
    };
  }

  /* ----------------------------- Generazione ----------------------------- */

  #sampling(opts) {
    const temperature = Number(opts.temperature ?? this.options.temperature);
    return {
      temperature,
      // `temp` è il nome usato dai SamplingParams nativi di Wllama, mentre
      // `temperature` è il campo OpenAI-compatibile: passiamo entrambi.
      temp: temperature,
      top_k: Number(opts.topK ?? this.options.topK),
      top_p: Number(opts.topP ?? this.options.topP),
      min_p: Number(opts.minP ?? this.options.minP),
      penalty_repeat: Number(opts.repeatPenalty ?? this.options.repeatPenalty),
      seed: Number(opts.seed ?? this.options.seed),
    };
  }

  /**
   * Esegue la generazione. Deve essere chiamata con il lock già acquisito.
   */
  async #complete(messages, opts) {
    await this.load();

    if (this._disposed) {
      throw new Error('GGUFEngine: istanza già rilasciata (dispose).');
    }

    if (this._backend === 'webgpu') {
      return this.#completeWebGPU(messages, opts);
    }

    const plan = await this.#buildPrompt(messages, opts);
    const { prompt, promptTokens, nPredict, stopStrings } = plan;
    const externalOnToken =
      typeof opts.onToken === 'function' ? opts.onToken : null;
    const externalSignal = opts.signal;

    this._busy = true;
    this._abortRequested = false;
    this.#setStatus('generating');

    const controller = new AbortController();
    this._abortController = controller;

    let emittedText = '';
    let tokenChunks = 0;
    let stopReached = false;
    let rawText = '';
    let usage = null;
    let timings = null;
    const startedAt = performance.now();

    const onAbort = () => {
      this._abortRequested = true;
      this.#emit('abort');
      controller.abort();
    };
    if (externalSignal?.aborted) onAbort();
    else externalSignal?.addEventListener('abort', onAbort, { once: true });

    // Wllama 3.x restituisce delta cumulativi? No: i chunk sono delta, quindi
    // accumuliamo noi e teniamo l'ultimo testo "pulito" per lo streaming.
    const consume = (fullRaw) => {
      if (stopReached) return;

      const cleanCurrent = stripSpecialTokens(fullRaw, stopStrings);
      if (cleanCurrent.length > emittedText.length) {
        const delta = cleanCurrent.startsWith(emittedText)
          ? cleanCurrent.slice(emittedText.length)
          : cleanCurrent.slice(commonPrefixLength(cleanCurrent, emittedText));
        emittedText = cleanCurrent;

        if (delta) {
          tokenChunks += 1;
          try {
            externalOnToken?.(delta, cleanCurrent);
          } catch (err) {
            console.warn('[GGUFEngine] onToken ha generato un errore:', err);
          }
          this.#emit('token', { delta, text: cleanCurrent, chunks: tokenChunks });
        }
      }

      if (findStopIndex(fullRaw, stopStrings) !== -1) {
        stopReached = true;
        controller.abort();
      }
    };

    try {
      if (plan.mode === 'chat') {
        const stream = await this._wllama.createChatCompletion({
          messages: plan.messages,
          max_tokens: nPredict,
          ...this.#sampling(opts),
          ...(stopStrings.length > 0 ? { stop: stopStrings } : {}),
          stream: true,
          abortSignal: controller.signal,
        });

        for await (const chunk of stream) {
          const delta = chunk?.choices?.[0]?.delta?.content;
          if (typeof delta === 'string' && delta) {
            rawText += delta;
            consume(rawText);
          }
          if (chunk?.usage) usage = chunk.usage;
          if (chunk?.timings) timings = chunk.timings;
          if (stopReached) break;
        }
      } else {
        const stream = await this._wllama.createCompletion({
          prompt,
          max_tokens: nPredict,
          ...this.#sampling(opts),
          ...(stopStrings.length > 0 ? { stop: stopStrings } : {}),
          stream: true,
          abortSignal: controller.signal,
        });

        for await (const chunk of stream) {
          const delta = chunk?.choices?.[0]?.text;
          if (typeof delta === 'string' && delta) {
            rawText += delta;
            consume(rawText);
          }
          if (chunk?.usage) usage = chunk.usage;
          if (chunk?.timings) timings = chunk.timings;
          if (stopReached) break;
        }
      }
    } catch (err) {
      // Un abort (utente o stop marker) è una conclusione normale: restituiamo
      // il testo prodotto finora. Gli altri errori vanno propagati.
      if (!controller.signal.aborted) {
        this.#emit('error', err);
        throw err;
      }
    } finally {
      externalSignal?.removeEventListener('abort', onAbort);
      this._abortController = null;
      this._busy = false;
    }

    const sanitized = stripSpecialTokens(rawText, stopStrings);
    const text = sanitized.length >= emittedText.length ? sanitized : emittedText;
    const durationMs = performance.now() - startedAt;
    // Con lo streaming il server può riportare usage parziale (a volte 1):
    // i delta realmente emessi sono la stima più affidabile per i tok/s.
    const completionTokens = tokenChunks || Number(usage?.completion_tokens) || 0;
    const result = {
      text,
      chunks: tokenChunks,
      durationMs,
      tokensPerSecond:
        durationMs > 0 ? (completionTokens / durationMs) * 1000 : 0,
      aborted: this._abortRequested,
      prompt,
      promptTokens: Number(usage?.prompt_tokens) || promptTokens,
      nPredict,
      contextSize: this.#contextSize(),
      template:
        plan.templateName === 'native'
          ? this._detected?.family || 'chat'
          : plan.templateName,
      backend: 'wasm',
      ...(usage ? { usage } : {}),
      ...(timings ? { timings } : {}),
    };

    this._lastResult = result;
    this.#emit('done', result);
    this._abortRequested = false;
    if (!this._disposed) this.#setStatus(this._ready ? 'ready' : 'idle');
    return result;
  }

  /**
   * Generazione via WebGPU/WebLLM. WebLLM applica da solo il template di chat
   * del modello: qui non si costruisce un prompt testuale né si tokenizza.
   */
  async #completeWebGPU(messages, opts) {
    const externalOnToken =
      typeof opts.onToken === 'function' ? opts.onToken : null;
    const externalSignal = opts.signal;
    const contextSize = this.#webgpuContextSize();
    const requested = Number(opts.maxTokens ?? this.options.maxTokens);
    const maxTokens =
      Number.isFinite(requested) && requested > 0 ? Math.floor(requested) : 256;

    const withSystem = this.#withSystem(
      messages,
      opts.systemPrompt !== undefined ? opts.systemPrompt : this.systemPrompt
    );
    const trimmed = this.#trimForContext(withSystem, contextSize, maxTokens);
    const stopStrings = this.#stopStringsFor(opts);
    const transcript = trimmed
      .map((m) => `${m.role}: ${m.content}`)
      .join('\n');

    this._busy = true;
    this._abortRequested = false;
    this.#setStatus('generating');

    let emittedText = '';
    let tokenChunks = 0;
    let stopReached = false;
    const startedAt = performance.now();

    const onAbort = () => {
      this._abortRequested = true;
      this.#emit('abort');
      this._gpu?.interrupt().catch(() => {});
    };
    if (externalSignal?.aborted) onAbort();
    externalSignal?.addEventListener('abort', onAbort, { once: true });

    try {
      const { text: rawText, usage } = await this._gpu.complete(trimmed, {
        temperature: Number(opts.temperature ?? this.options.temperature),
        topP: Number(opts.topP ?? this.options.topP),
        topK: Number(opts.topK ?? this.options.topK),
        maxTokens,
        seed: Number(opts.seed ?? this.options.seed),
        stop: stopStrings,
        signal: externalSignal,
        shouldStop: () => this._abortRequested || stopReached,
        onToken: (delta, fullText) => {
          if (stopReached) return;
          const cleanFull = stripSpecialTokens(fullText, stopStrings);
          if (cleanFull.length > emittedText.length) {
            const cleanDelta = cleanFull.startsWith(emittedText)
              ? cleanFull.slice(emittedText.length)
              : cleanFull.slice(commonPrefixLength(cleanFull, emittedText));
            emittedText = cleanFull;

            if (cleanDelta) {
              tokenChunks += 1;
              try {
                externalOnToken?.(cleanDelta, cleanFull);
              } catch (err) {
                console.warn('[GGUFEngine] onToken ha generato un errore:', err);
              }
              this.#emit('token', {
                delta: cleanDelta,
                text: cleanFull,
                chunks: tokenChunks,
              });
            }
          }

          if (findStopIndex(fullText, stopStrings) !== -1) {
            stopReached = true;
          }
        },
      });

      const sanitized = stripSpecialTokens(rawText, stopStrings);
      const text = sanitized.length >= emittedText.length ? sanitized : emittedText;
      const durationMs = performance.now() - startedAt;
      const completionTokens = Number(usage?.completion_tokens);
      const promptTokens = Number(usage?.prompt_tokens);
      const chunks =
        Number.isFinite(completionTokens) && completionTokens > 0
          ? completionTokens
          : tokenChunks;

      const result = {
        text,
        chunks,
        durationMs,
        tokensPerSecond: durationMs > 0 ? (chunks / durationMs) * 1000 : 0,
        aborted: this._abortRequested,
        prompt: transcript,
        promptTokens:
          Number.isFinite(promptTokens) && promptTokens > 0
            ? promptTokens
            : estimateTokens(transcript),
        nPredict: maxTokens,
        contextSize,
        template: 'webllm',
        backend: 'webgpu',
        model: this._activeModelUrl,
      };

      this._lastResult = result;
      this.#emit('done', result);
      return result;
    } catch (err) {
      this.#emit('error', err);
      throw err;
    } finally {
      externalSignal?.removeEventListener('abort', onAbort);
      this._busy = false;
      this._abortRequested = false;
      if (!this._disposed) this.#setStatus(this._ready ? 'ready' : 'idle');
    }
  }

  /**
   * Genera una risposta a partire da un singolo prompt.
   * @returns {Promise<string>}
   */
  async generate(prompt, opts = {}) {
    const result = await this.generateDetailed(prompt, opts);
    return result.text;
  }

  /** Alias di generate(). */
  ask(prompt, opts = {}) {
    return this.generate(prompt, opts);
  }

  /** Come generate(), ma restituisce anche statistiche e prompt formattato. */
  async generateDetailed(prompt, opts = {}) {
    if (typeof prompt !== 'string' || !prompt.trim()) {
      throw new TypeError('GGUFEngine: "prompt" deve essere una stringa non vuota.');
    }
    const messages = this.#normalizeMessages([{ role: 'user', content: prompt }]);
    const release = await this.#acquire();
    try {
      return await this.#complete(messages, opts);
    } finally {
      release();
    }
  }

  /**
   * Chat stateless: passi l'intera conversazione ad ogni chiamata.
   * @param {{role: string, content: string}[]} messages
   * @returns {Promise<string>}
   */
  async chat(messages, opts = {}) {
    const result = await this.chatDetailed(messages, opts);
    return result.text;
  }

  /** Come chat(), ma restituisce anche statistiche e prompt formattato. */
  async chatDetailed(messages, opts = {}) {
    const normalized = this.#normalizeMessages(messages);
    const release = await this.#acquire();
    try {
      return await this.#complete(normalized, opts);
    } finally {
      release();
    }
  }

  /** Streaming di un singolo prompt: async iterable di delta di testo. */
  stream(prompt, opts = {}) {
    if (typeof prompt !== 'string' || !prompt.trim()) {
      throw new TypeError('GGUFEngine: "prompt" deve essere una stringa non vuota.');
    }
    return this.#streamMessages(
      this.#normalizeMessages([{ role: 'user', content: prompt }]),
      opts
    );
  }

  /** Streaming di una conversazione completa. */
  chatStream(messages, opts = {}) {
    return this.#streamMessages(this.#normalizeMessages(messages), opts);
  }

  async *#streamMessages(messages, opts) {
    const release = await this.#acquire();
    const queue = [];
    let finished = false;
    let failure = null;
    let wake = null;

    const notify = () => {
      if (wake) {
        const resolve = wake;
        wake = null;
        resolve();
      }
    };

    const task = this.#complete(messages, {
      ...opts,
      onToken: (delta, text) => {
        queue.push(delta);
        opts.onToken?.(delta, text);
        notify();
      },
    })
      .then((result) => {
        this._lastResult = result;
        return result;
      })
      .catch((err) => {
        failure = err;
        return null;
      })
      .finally(() => {
        finished = true;
        notify();
      });

    try {
      while (!finished || queue.length > 0) {
        if (queue.length > 0) {
          yield queue.shift();
          continue;
        }
        if (finished) break;
        await new Promise((resolve) => {
          wake = resolve;
        });
      }

      const result = await task;
      if (failure) throw failure;
      return result;
    } finally {
      if (!finished) {
        // Il consumatore ha interrotto l'iterazione: ferma la generazione.
        this.abort();
        await task.catch(() => {});
      }
      release();
    }
  }

  /* --------------------------- Modalità stateful --------------------------- */

  /**
   * Invia un messaggio mantenendo la cronologia interna.
   * @returns {Promise<string>}
   */
  async send(message, opts = {}) {
    if (typeof message !== 'string' || !message.trim()) {
      throw new TypeError('GGUFEngine.send: il messaggio deve essere una stringa non vuota.');
    }

    const userMessage = { role: 'user', content: message };
    const conversation = [...this.messages, userMessage];
    const result = await this.chatDetailed(conversation, opts);

    this.messages = [
      ...conversation,
      { role: 'assistant', content: result.text },
    ];
    this.#trimInternalHistory();
    return result.text;
  }

  /** Versione streaming di send(): aggiorna la cronologia a fine risposta. */
  async *sendStream(message, opts = {}) {
    if (typeof message !== 'string' || !message.trim()) {
      throw new TypeError('GGUFEngine.sendStream: il messaggio deve essere una stringa non vuota.');
    }

    const conversation = [...this.messages, { role: 'user', content: message }];
    let answer = '';

    for await (const delta of this.#streamMessages(conversation, opts)) {
      answer += delta;
      yield delta;
    }

    this.messages = [
      ...conversation,
      { role: 'assistant', content: answer },
    ];
    this.#trimInternalHistory();
  }

  /** Svuota la cronologia della modalità stateful. */
  resetHistory(systemPrompt) {
    this.messages = [];
    if (arguments.length > 0) this.systemPrompt = systemPrompt;
    return this;
  }

  setSystemPrompt(systemPrompt) {
    this.systemPrompt = systemPrompt;
    this.options.systemPrompt = systemPrompt;
    return this;
  }

  getHistory() {
    return this.messages.map((m) => ({ ...m }));
  }

  #trimInternalHistory() {
    const contextSize = this.isWebGPU
      ? this.#webgpuContextSize()
      : this.#contextSize();
    const budget = contextSize - (Number(this.options.maxTokens) || 0) - 32;
    if (budget <= 0) return;

    const total = () =>
      estimateTokens(
        [this.systemPrompt, ...this.messages.map((m) => m.content)].join('\n')
      );

    while (total() > budget && this.messages.length > 2) {
      this.messages.splice(0, 2);
    }
  }

  /* ------------------------------- Controllo ------------------------------- */

  /** Interrompe la generazione in corso (se presente). */
  abort() {
    if (!this._busy) return false;
    this._abortRequested = true;
    this.#emit('abort');
    this._abortController?.abort();
    if (this._backend === 'webgpu') {
      this._gpu?.interrupt().catch(() => {});
    }
    return true;
  }

  /** Rilascia la sessione Wllama e libera la memoria. */
  async dispose() {
    if (this._disposed) return;

    this._disposed = true;
    this.abort();

    // Attende l'eventuale generazione in corso prima di liberare il motore.
    const release = await this.#acquire();
    try {
      if (this._wllama) {
        try {
          await this._wllama.exit();
        } catch (err) {
          console.warn('[GGUFEngine] errore durante exit():', err);
        }
      }
      if (this._gpu) {
        try {
          await this._gpu.dispose();
        } catch (err) {
          console.warn('[GGUFEngine] errore durante il rilascio GPU:', err);
        }
      }
    } finally {
      this._wllama = null;
      this._gpu = null;
      this._backend = null;
      this._ready = false;
      this._loadPromise = null;
      this._busy = false;
      this._lastResult = null;
      release();
      this.#setStatus('disposed');
    }
  }

  /** Alias di dispose(). */
  unload() {
    return this.dispose();
  }

  /** Path WASM di default (utile per self-hosting). */
  static get defaultWasmPaths() {
    return { ...DEFAULT_WASM_PATHS };
  }

  /** Worker/WASM compat self-hosted per Safari e browser senza JSPI/MEMORY64. */
  static get defaultWllamaCompatPaths() {
    return { ...DEFAULT_WLLAMA_COMPAT_PATHS };
  }

  static get defaultModuleUrl() {
    return DEFAULT_WLLAMA_MODULE;
  }

  /** Modulo WebLLM self-hosted (se presente in vendor/web-llm). */
  static get defaultWebLLMModuleUrl() {
    return DEFAULT_WEBLLM_VENDOR_MODULE;
  }

  /** Runtime web usato quando gli asset locali in vendor/ non sono disponibili. */
  static get cdnModuleUrl() {
    return CDN_WLLAMA_URLS.moduleUrl;
  }

  static get cdnWasmPaths() {
    return { ...CDN_WLLAMA_URLS.wasmPaths };
  }

  static get cdnCompatPaths() {
    return { ...CDN_WLLAMA_URLS.compat };
  }

  /**
   * Azzera le verifiche di esistenza già effettuate (utile se gli asset
   * vengono aggiunti/rimossi dal server senza ricaricare la pagina).
   */
  static invalidateAssetCache() {
    assetAvailability.clear();
    runtimeSourcesCache.clear();
  }
}

/**
 * Factory di comodo.
 * @param {ConstructorParameters<typeof GGUFEngine>[0]} config
 */
export function createEngine(config) {
  return new GGUFEngine(config);
}

/** Utility WebGPU ri-esportate per comodità dei consumer. */
export {
  WebGPUBackend,
  describeAdapter,
  DEFAULT_WEBLLM_MODEL,
  DEFAULT_WEBLLM_MODULE_URL,
} from './webgpu-backend.js';

export default GGUFEngine;
