/**
 * ============================================================================
 * WebGPUBackend — accelerazione GPU (WebGPU) tramite WebLLM / MLC
 * ============================================================================
 * Wllama è un runtime WASM/CPU: non espone alcun backend GPU. Per usare la GPU
 * del dispositivo nel browser questo modulo integra
 * [`@mlc-ai/web-llm`](https://github.com/mlc-ai/web-llm), che esegue modelli
 * MLC precompilati su WebGPU.
 *
 * Conseguenze importanti:
 *
 *  - i modelli WebLLM **non sono GGUF**: il backend GPU scarica un modello MLC
 *    (es. "LFM2.5-1.2B-Instruct-q4f32_1-MLC") la prima volta e poi lo riusa
 *    dalla cache del browser;
 *  - il backend GPU è complementare a Wllama: `GGUFEngine` sceglie l'uno o
 *    l'altro e ripiega automaticamente sul GGUF/WASM quando WebGPU non c'è;
 *  - WebGPU richiede un contesto sicuro (https:// oppure localhost) e un
 *    browser con `navigator.gpu` (Chrome/Edge recenti, Safari 18+).
 *
 * Il modulo non ha dipendenze npm obbligatorie: il runtime WebLLM viene
 * importato dinamicamente dal CDN (o da un file locale in vendor/web-llm/).
 *
 * Uso diretto:
 *
 *   import { WebGPUBackend } from './webgpu-backend.js';
 *
 *   const info = await WebGPUBackend.detect();
 *   if (!info.supported) console.log(info.reason);
 *
 *   const backend = new WebGPUBackend({ model: 'LFM2.5-1.2B-Instruct-q4f32_1-MLC' });
 *   await backend.load({ onProgress: ({ progress, text }) => console.log(text) });
 *
 *   for await (const { delta } of backend.stream(messages)) process.stdout.write(delta);
 *   await backend.dispose();
 * ============================================================================
 */

/** Runtime WebLLM usato quando non viene fornito un path locale. */
export const DEFAULT_WEBLLM_MODULE_URL = 'https://esm.run/@mlc-ai/web-llm';

/**
 * Modello GPU di esempio: piccolo, veloce da scaricare e adatto anche a GPU
 * integrate. È solo un default: `GGUFEngine` accetta qualsiasi id MLC.
 */
export const DEFAULT_WEBLLM_MODEL = 'LFM2.5-1.2B-Instruct-q4f32_1-MLC';

/** Feature WebGPU necessarie ai modelli f16/f32 più comuni. */
const OPTIONAL_FEATURES = ['shader-f16', 'float32-filterable'];

/** Cache del rilevamento: requestAdapter() è asincrono e relativamente lento. */
let detectionCache = null;

/* ---------------------------------------------------------------------------
 * Rilevamento
 * ------------------------------------------------------------------------- */

function hasGpuApi() {
  return Boolean(globalThis.navigator?.gpu);
}

function secureContextStatus() {
  const loc = globalThis.location;
  if (!loc) return { secure: true, reason: '' };
  if (globalThis.isSecureContext === true) return { secure: true, reason: '' };

  const host = String(loc.hostname || '').toLowerCase();
  const isLocal =
    host === 'localhost' ||
    host === '127.0.0.1' ||
    host === '[::1]' ||
    host.endsWith('.localhost');

  if (loc.protocol === 'https:' || isLocal) return { secure: true, reason: '' };
  return {
    secure: false,
    reason:
      'WebGPU richiede un contesto sicuro: usa https:// oppure http://localhost.',
  };
}

function readAdapterInfo(adapter, source = adapter?.info || null) {
  // `adapter.info` è un GPUAdapterInfo con getter sul prototipo: va copiato
  // campo per campo (lo spread di un oggetto con getter produce spesso {}).
  const info = source
    ? {
        vendor: source.vendor || '',
        architecture: source.architecture || '',
        device: source.device || '',
        description: source.description || '',
      }
    : { vendor: '', architecture: '', device: '', description: '' };

  if (typeof adapter?.isFallbackAdapter === 'boolean') {
    info.fallbackAdapter = adapter.isFallbackAdapter;
  }
  return info;
}

/** Etichetta leggibile per l'adattatore (usata dalla UI e nei log). */
export function describeAdapter(info) {
  if (!info) return 'WebGPU';
  const parts = [info.vendor, info.architecture, info.device, info.description]
    .map((value) => String(value || '').trim())
    .filter(Boolean);
  if (parts.length === 0) return info.fallbackAdapter ? 'WebGPU (software)' : 'WebGPU';
  return `WebGPU · ${parts.join(' ')}`;
}

/** L'adattatore supporta il formato f16 (richiesto dai modelli "...-q4f16_1")? */
export function supportsF16(info) {
  return Boolean(info?.features?.includes('shader-f16'));
}

/* ---------------------------------------------------------------------------
 * WebGPUBackend
 * ------------------------------------------------------------------------- */

export class WebGPUBackend {
  /**
   * Rileva WebGPU senza lanciare eccezioni.
   *
   * @param {{force?: boolean, powerPreference?: 'low-power'|'high-performance'}} [options]
   * @returns {Promise<{
   *   supported: boolean, reason: string, secureContext: boolean,
   *   adapterInfo: {vendor:string,architecture:string,device:string,description:string,fallbackAdapter?:boolean}|null,
   *   features: string[], limits: {maxBufferSize?:number,maxStorageBufferBindingSize?:number}|null
   * }>}
   */
  static async detect({ force = false, powerPreference = 'high-performance' } = {}) {
    if (!force && detectionCache) return detectionCache;

    detectionCache = (async () => {
      const secure = secureContextStatus();
      const base = {
        supported: false,
        reason: '',
        secureContext: secure.secure,
        adapterInfo: null,
        features: [],
        limits: null,
      };

      if (!hasGpuApi()) {
        return {
          ...base,
          reason: secure.secure
            ? 'navigator.gpu non disponibile: questo browser non espone WebGPU.'
            : secure.reason,
        };
      }
      if (!secure.secure) return { ...base, reason: secure.reason };

      let adapter = null;
      try {
        adapter = await globalThis.navigator.gpu.requestAdapter({ powerPreference });
      } catch (err) {
        return {
          ...base,
          reason: `Richiesta dell'adattatore WebGPU fallita: ${err?.message || err}`,
        };
      }

      if (!adapter) {
        return {
          ...base,
          reason:
            'Nessun adattatore WebGPU disponibile: GPU non supportata, driver bloccato o GPU occupata da un altro processo.',
        };
      }

      let adapterSource = adapter.info || null;
      if (!adapterSource && typeof adapter.requestAdapterInfo === 'function') {
        // API deprecata (Chrome < 128): è asincrona.
        try {
          adapterSource = await adapter.requestAdapterInfo();
        } catch {
          adapterSource = null;
        }
      }

      const adapterInfo = readAdapterInfo(adapter, adapterSource);
      const features = adapter.features ? [...adapter.features] : [];
      const limits = adapter.limits
        ? {
            maxBufferSize: adapter.limits.maxBufferSize,
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
          }
        : null;

      return {
        supported: true,
        reason: '',
        secureContext: true,
        adapterInfo,
        features,
        limits,
      };
    })();

    return detectionCache;
  }

  /** True se `navigator.gpu` esiste (non implica un adattatore disponibile). */
  static isSupported() {
    return hasGpuApi();
  }

  /** Azzera il rilevamento cacheato (utile dopo un cambio di GPU/permissioni). */
  static invalidateDetection() {
    detectionCache = null;
  }

  /**
   * @param {Object} config
   * @param {string} config.model                 Id del modello MLC (obbligatorio).
   * @param {Object} [config.modelRecord]          ModelRecord custom (self-hosting) da aggiungere a model_list.
   * @param {string} [config.moduleUrl]            Modulo ESM di WebLLM (default: CDN).
   * @param {number} [config.contextSize=4096]     Contesto dichiarato (per il trimming della cronologia).
   */
  constructor(config = {}) {
    const model = config.model || config.webgpuModel || '';
    if (!model || typeof model !== 'string') {
      throw new TypeError(
        'WebGPUBackend: "model" è obbligatorio (id del modello MLC, es. "LFM2.5-1.2B-Instruct-q4f32_1-MLC").'
      );
    }

    this.model = model;
    this.modelRecord = config.modelRecord || null;
    this.moduleUrl = config.moduleUrl || DEFAULT_WEBLLM_MODULE_URL;
    this.contextSize = Number(config.contextSize) > 0 ? Number(config.contextSize) : 4096;

    this._engine = null;
    this._module = null;
    this._info = null;
    this._loading = null;
    this._lastUsage = null;
  }

  get modelId() {
    return this.model;
  }

  /** Istanza WebLLM (per uso avanzato), o null prima di load(). */
  get engine() {
    return this._engine;
  }

  /** Info sull'adattatore GPU (null prima di load()). */
  get info() {
    return this._info;
  }

  /** Ultimo `usage` restituito da WebLLM (token di prompt/completamento). */
  get lastUsage() {
    return this._lastUsage;
  }

  get isLoaded() {
    return this._engine != null;
  }

  /** Contesto effettivo dichiarato: usato da GGUFEngine per il trimming. */
  get contextWindow() {
    return this.contextSize;
  }

  /**
   * Carica il runtime WebLLM e il modello. Idempotente.
   *
   * @param {{onProgress?: (p: {progress:number, text:string})=>void, signal?: AbortSignal}} [options]
   * @returns {Promise<WebGPUBackend>}
   */
  async load({ onProgress, signal } = {}) {
    if (this._engine) return this;
    if (this._loading) return this._loading;

    this._loading = (async () => {
      const detection = await WebGPUBackend.detect();
      if (!detection.supported) {
        throw gpuError(new Error(detection.reason || 'WebGPU non disponibile.'));
      }
      this._info = detection;

      const mod = await import(/* @vite-ignore */ this.moduleUrl);
      this._module = mod;

      const CreateMLCEngine =
        mod.CreateMLCEngine ||
        mod.default?.CreateMLCEngine ||
        mod.webllm?.CreateMLCEngine;
      const MLCEngine = mod.MLCEngine || mod.default?.MLCEngine;

      const initProgressCallback = (report) => {
        if (!onProgress) return;
        const progress = Number(report?.progress);
        onProgress({
          progress: Number.isFinite(progress)
            ? Math.min(1, Math.max(0, progress))
            : 0,
          text: String(report?.text || 'Caricamento del modello GPU…'),
        });
      };

      const engineConfig = { initProgressCallback };
      if (this.modelRecord) {
        const prebuilt = mod.prebuiltAppConfig;
        const records = Array.isArray(prebuilt?.model_list)
          ? prebuilt.model_list
          : [];
        engineConfig.appConfig = {
          ...(prebuilt || {}),
          model_list: [
            ...records.filter(
              (record) => record?.model_id !== this.modelRecord.model_id
            ),
            this.modelRecord,
          ],
        };
      }

      const onAbort = () => {
        this.interrupt().catch(() => {});
      };
      if (signal?.aborted) throw abortError();
      signal?.addEventListener('abort', onAbort, { once: true });

      try {
        if (typeof CreateMLCEngine === 'function') {
          this._engine = await CreateMLCEngine(this.model, engineConfig);
        } else if (typeof MLCEngine === 'function') {
          const engine = new MLCEngine(engineConfig);
          await engine.reload(this.model);
          this._engine = engine;
        } else {
          throw new Error(
            `Il modulo "${this.moduleUrl}" non esporta CreateMLCEngine/MLCEngine.`
          );
        }
      } catch (err) {
        this._engine = null;
        throw gpuError(err);
      } finally {
        signal?.removeEventListener('abort', onAbort);
      }

      return this;
    })().finally(() => {
      this._loading = null;
    });

    return this._loading;
  }

  /** Traduce le opzioni del motore in una richiesta chat di WebLLM. */
  #chatRequest(messages, opts = {}) {
    const request = { messages };
    const temperature = Number(opts.temperature);
    const topP = Number(opts.topP);
    const topK = Number(opts.topK);
    const maxTokens = Number(opts.maxTokens);
    const seed = Number(opts.seed);

    if (Number.isFinite(temperature)) request.temperature = temperature;
    if (Number.isFinite(topP)) request.top_p = topP;
    if (Number.isFinite(topK) && topK > 0) request.top_k = topK;
    if (Number.isFinite(maxTokens) && maxTokens > 0) {
      request.max_tokens = Math.floor(maxTokens);
    }
    if (Number.isFinite(seed) && seed >= 0) request.seed = Math.floor(seed);
    if (Array.isArray(opts.stop) && opts.stop.length) {
      request.stop = opts.stop.filter((piece) => typeof piece === 'string' && piece);
    }
    return request;
  }

  /** Interrompe la generazione in corso (WebLLM la ferma al token successivo). */
  async interrupt() {
    if (!this._engine) return;
    try {
      await this._engine.interruptGenerate();
    } catch {
      /* nessuna generazione attiva */
    }
  }

  /**
   * Streaming dei delta di testo.
   * Imposta `lastUsage` quando WebLLM restituisce il blocco `usage` finale.
   *
   * @param {{role:string, content:string}[]} messages
   * @param {Object} [opts] temperature, topP, topK, maxTokens, seed, stop, signal, shouldStop
   * @returns {AsyncGenerator<{delta: string}>}
   */
  async *stream(messages, opts = {}) {
    if (!this._engine) {
      throw new Error('WebGPUBackend: chiama load() prima di generare.');
    }

    const request = {
      ...this.#chatRequest(messages, opts),
      stream: true,
      stream_options: { include_usage: true },
    };

    const onAbort = () => {
      this.interrupt().catch(() => {});
    };
    if (opts.signal?.aborted) throw abortError();
    opts.signal?.addEventListener('abort', onAbort, { once: true });

    this._lastUsage = null;
    let completion = null;
    let interrupted = false;
    try {
      completion = await this._engine.chat.completions.create(request);
      for await (const chunk of completion) {
        if (opts.shouldStop?.() || opts.signal?.aborted) {
          interrupted = true;
          await this.interrupt();
          break;
        }
        if (chunk?.usage) this._lastUsage = chunk.usage;
        const choice = chunk?.choices?.[0];
        const delta = choice?.delta?.content ?? choice?.text ?? '';
        if (delta) yield { delta };
      }
    } finally {
      opts.signal?.removeEventListener('abort', onAbort);
      // Se il consumatore abbandona l'iterazione, ferma la generazione.
      if (!interrupted && (opts.shouldStop?.() || opts.signal?.aborted)) {
        await this.interrupt();
      }
    }
  }

  /**
   * Generazione non-streaming (accumula il testo e chiama `onToken`).
   *
   * @returns {Promise<{text: string, usage: Object|null}>}
   */
  async complete(messages, opts = {}) {
    let text = '';
    for await (const { delta } of this.stream(messages, opts)) {
      text += delta;
      try {
        opts.onToken?.(delta, text);
      } catch (err) {
        console.warn('[WebGPUBackend] onToken ha generato un errore:', err);
      }
    }
    return { text, usage: this._lastUsage };
  }

  /** Scarica il modello dalla GPU e libera le risorse WebLLM. */
  async dispose() {
    this._loading = null;
    const engine = this._engine;
    this._engine = null;
    this._lastUsage = null;
    if (!engine) return;
    try {
      await engine.unload();
    } catch (err) {
      console.warn('[WebGPUBackend] errore durante unload():', err);
    }
  }
}

/* ---------------------------------------------------------------------------
 * Errori
 * ------------------------------------------------------------------------- */

function abortError() {
  const err = new Error('Generazione interrotta.');
  err.name = 'AbortError';
  return err;
}

/** Rende leggibili gli errori tipici di WebGPU/WebLLM. */
export function gpuError(err) {
  const raw = err?.message || String(err || 'Errore WebGPU sconosciuto.');
  let message = raw;

  if (/shader-f16|float16|f16/i.test(raw)) {
    message =
      'La GPU non supporta lo shader f16 richiesto da questo modello MLC. ' +
      'Scegli una variante "-q4f32_1" oppure usa il backend CPU (GGUF/WASM). ' +
      `Dettaglio: ${raw}`;
  } else if (/out of memory|\boom\b|device lost|buffer size|maxstoragebufferbindingsize/i.test(raw)) {
    message =
      'Memoria GPU insufficiente per questo modello. Scegli un modello MLC più piccolo ' +
      '(es. una variante 0.5B) o torna al backend CPU (GGUF/WASM). ' +
      `Dettaglio: ${raw}`;
  } else if (/failed to fetch|networkerror|load failed|importing module/i.test(raw)) {
    message =
      'Impossibile scaricare il runtime WebLLM o il modello GPU. Verifica la connessione, ' +
      'i blocchi CORS e che l\'URL del modulo WebLLM sia raggiungibile. ' +
      `Dettaglio: ${raw}`;
  } else if (/not found|404|cannot find model|model_id/i.test(raw)) {
    message =
      'Modello GPU non trovato nell\'elenco MLC precompilato. Controlla l\'id (es. ' +
      '"LFM2.5-1.2B-Instruct-q4f32_1-MLC") o fornisci un modelRecord personalizzato. ' +
      `Dettaglio: ${raw}`;
  } else if (/webgpu|adapter|navigator\.gpu/i.test(raw)) {
    message =
      'WebGPU non è utilizzabile in questo contesto. Controlla che il browser la supporti, ' +
      'che la pagina sia servita da https:// o localhost e che la GPU non sia bloccata. ' +
      `Dettaglio: ${raw}`;
  }

  const wrapped = new Error(message);
  wrapped.name = 'WebGPULoadError';
  wrapped.cause = err;
  return wrapped;
}

/** Feature note, esposte per diagnostica nella UI. */
export function summarizeFeatures(features = []) {
  return OPTIONAL_FEATURES.map((name) => ({
    name,
    available: features.includes(name),
  }));
}

export default WebGPUBackend;
