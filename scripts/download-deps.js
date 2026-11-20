#!/usr/bin/env node
/**
 * ============================================================================
 * download-deps.js — scarica tutte le dipendenze runtime sul server
 * ============================================================================
 * Dopo l'esecuzione l'applicazione non contatta più CDN o Hugging Face:
 *
 *   vendor/wllama/esm/index.js                    runtime @wllama/wllama
 *   vendor/wllama/esm/wasm/wllama.wasm            WASM (v3, JSPI/MEMORY64)
 *   vendor/wllama-compat/wasm/wllama.js           compat worker (Safari/ecc.)
 *   vendor/wllama-compat/wasm/wllama.wasm         compat WASM
 *   vendor/web-llm/lib/index.js                   runtime WebLLM (backend GPU)
 *   models/MNN-Pocket-0.3B-Instruct-Q4_1.gguf      modello GGUF
 *
 * Entrambi i runtime vengono scaricati per default: il backend GPU può così
 * usare `./vendor/web-llm/lib/index.js` senza mai toccare la rete.
 *
 * Uso:
 *   bun run deps
 *   node scripts/download-deps.js
 *   bash scripts/download-deps.sh          (equivalente, senza Node)
 *
 * Opzioni:
 *   --force                 riscarica anche ciò che è già presente
 *   --skip-wllama           non scarica Wllama (runtime CPU)
 *   --skip-webllm           non scarica WebLLM (runtime GPU)
 *   --skip-runtime          non scarica nessuno dei due runtime
 *   --skip-model            non scarica il modello GGUF
 *   --check                 mostra solo lo stato dei file, senza scaricare
 *   --with-webllm           compatibilità: WebLLM è già il default (no-op)
 *   --wllama-version=3.13.0 versione dei pacchetti Wllama e Wllama compat
 *   --webllm-version=0.2.89 versione del pacchetto WebLLM
 *   --model-url=URL         URL alternativo del modello
 *   --model-file=NOME.gguf  nome file locale del modello
 *
 * I flag booleani accettano anche la forma --flag=1 / --flag=true / --flag=0.
 *
 * Variabili d'ambiente equivalenti:
 *   WLLAMA_VERSION, WLLAMA_TARBALL, WLLAMA_INTEGRITY,
 *   WLLAMA_COMPAT_VERSION, WLLAMA_COMPAT_TARBALL, WLLAMA_COMPAT_INTEGRITY,
 *   WEBLLM_VERSION, WEBLLM_TARBALL, WEBLLM_INTEGRITY,
 *   MODEL_URL, MODEL_FILE, MODEL_SHA256,
 *   FORCE=1, SKIP_WLLAMA=1, SKIP_WEBLLM=1, SKIP_MODEL=1
 *
 * Il download del modello è riprendibile: se viene interrotto, rilanciando il
 * comando riparte dal byte successivo (HTTP Range).
 * ============================================================================
 */
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import {
  mkdir,
  open,
  rename,
  rm,
  stat,
  writeFile,
} from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Readable } from 'node:stream';
import { gunzipSync } from 'node:zlib';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));

const DEFAULT_MODEL_URL =
  'https://huggingface.co/taobao-mnn/MNN-Pocket-0.3B-GGUF/resolve/main/MNN-Pocket-0.3B-Instruct-Q4_1.gguf';

const VENDOR_ROOT = join(ROOT, 'vendor');
const WLLAMA_DIR = join(VENDOR_ROOT, 'wllama');
const WLLAMA_COMPAT_DIR = join(VENDOR_ROOT, 'wllama-compat');
const WEBLLM_DIR = join(VENDOR_ROOT, 'web-llm');
const MODEL_DIR = join(ROOT, 'models');
const REGISTRY_URL = 'https://registry.npmjs.org';

/* ---------------------------------------------------------------------------
 * CLI / configurazione
 * ------------------------------------------------------------------------- */

const TRUE_VALUES = new Set(['1', 'true', 'yes', 'on', '']);
const FALSE_VALUES = new Set(['0', 'false', 'no', 'off']);

function parseArgs(argv) {
  const args = { _: [] };
  for (const raw of argv) {
    if (!raw.startsWith('--')) {
      args._.push(raw);
      continue;
    }
    const body = raw.slice(2);
    const eq = body.indexOf('=');
    const key = eq === -1 ? body : body.slice(0, eq);
    const value = eq === -1 ? true : body.slice(eq + 1);
    args[key] = value;
  }
  return args;
}

/**
 * Interpreta un flag booleano in tutte le forme accettate.
 * Un flag assente vale `fallback`.
 */
function boolFlag(args, key, envName, fallback = false) {
  const raw = args[key];
  if (raw === true || raw === '') return true;
  if (typeof raw === 'string') {
    const value = raw.trim().toLowerCase();
    if (TRUE_VALUES.has(value)) return true;
    if (FALSE_VALUES.has(value)) return false;
  }
  const env = String(process.env[envName] ?? '').trim().toLowerCase();
  if (env) {
    if (TRUE_VALUES.has(env)) return true;
    if (FALSE_VALUES.has(env)) return false;
  }
  return fallback;
}

function strFlag(args, key, envName, fallback) {
  const raw = args[key];
  const value =
    typeof raw === 'string' && raw.trim() !== ''
      ? raw
      : String(process.env[envName] ?? '').trim() || fallback;
  return String(value).trim();
}

const args = parseArgs(process.argv.slice(2));

const KNOWN_KEYS = new Set([
  'help', 'h',
  'force',
  'skip-runtime', 'skip-wllama', 'skip-webllm', 'skip-model',
  'check', 'with-webllm',
  'wllama-version', 'webllm-version', 'model-url', 'model-file',
]);

if (args.help === true || args.h === true) {
  const help = (await import('node:fs')).readFileSync(
    fileURLToPath(import.meta.url),
    'utf8'
  )
    .split('\n')
    .filter((line, index) => index > 0 && line.trimStart().startsWith('*'))
    .map((line) => line.replace(/^\s*\* ?/, ''))
    .join('\n');
  log(help.trimEnd());
  process.exit(0);
}

const unknown = Object.keys(args).filter(
  (key) => key !== '_' && !KNOWN_KEYS.has(key)
);
if (unknown.length > 0) {
  log(`❌ Opzione sconosciuta: ${unknown.map((k) => `--${k}`).join(', ')}`);
  log('  Usa --help per l\'elenco delle opzioni.');
  process.exit(2);
}

const FORCE = boolFlag(args, 'force', 'FORCE');
const SKIP_ALL_RUNTIME = boolFlag(args, 'skip-runtime', 'SKIP_RUNTIME');
const SKIP_WLLAMA = SKIP_ALL_RUNTIME || boolFlag(args, 'skip-wllama', 'SKIP_WLLAMA');
const SKIP_WEBLLM = SKIP_ALL_RUNTIME || boolFlag(args, 'skip-webllm', 'SKIP_WEBLLM');
const SKIP_MODEL = boolFlag(args, 'skip-model', 'SKIP_MODEL');
const CHECK_ONLY = boolFlag(args, 'check', 'CHECK');

if (boolFlag(args, 'with-webllm', 'WITH_WEBLLM')) {
  // WebLLM è il default da questa versione: il flag resta accettato per non
  // rompere script e CI esistenti.
  if (SKIP_ALL_RUNTIME) {
    log('⚠ --with-webllm ignorato: --skip-runtime ha la precedenza.');
  }
}

const WLLAMA_VERSION = strFlag(args, 'wllama-version', 'WLLAMA_VERSION', '3.13.0');
const WEBLLM_VERSION = strFlag(args, 'webllm-version', 'WEBLLM_VERSION', '0.2.89');

function fileNameFromUrl(url) {
  try {
    return decodeURIComponent(new URL(url).pathname.split('/').pop() || '');
  } catch {
    return basename(String(url).split('?')[0]);
  }
}

const MODEL_URL = strFlag(args, 'model-url', 'MODEL_URL', DEFAULT_MODEL_URL);
const MODEL_FILE = basename(
  strFlag(
    args,
    'model-file',
    'MODEL_FILE',
    fileNameFromUrl(MODEL_URL) || 'model.gguf'
  )
);
const MODEL_PATH = join(MODEL_DIR, MODEL_FILE);
const MODEL_PART_PATH = `${MODEL_PATH}.part`;
const MODEL_META_PATH = `${MODEL_PATH}.json`;
const MODEL_SHA256 = String(process.env.MODEL_SHA256 || '')
  .trim()
  .toLowerCase();

/* ---------------------------------------------------------------------------
 * Utility
 * ------------------------------------------------------------------------- */

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return '?';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit++;
  }
  return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '—';
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${String(s % 60).padStart(2, '0')}s`;
}

async function fileSize(path) {
  try {
    return (await stat(path)).size;
  } catch {
    return null;
  }
}

async function exists(path) {
  return (await fileSize(path)) !== null;
}

async function atomicWrite(path, data) {
  await mkdir(dirname(path), { recursive: true });
  const tmp = `${path}.tmp-${process.pid}`;
  await writeFile(tmp, data);
  await rename(tmp, path);
}

function log(message = '') {
  process.stdout.write(`${message}\n`);
}

function logStep(message) {
  log(`\n▶ ${message}`);
}

function logOk(message) {
  log(`  ✓ ${message}`);
}

function logSkip(message) {
  log(`  • ${message}`);
}

/**
 * fetch con timeout, retry ed errori leggibili.
 * Il timeout vale solo per la ricezione degli header: il body può continuare
 * a lungo senza essere interrotto.
 */
async function request(url, options = {}, { retries = 3, timeoutMs = 60_000 } = {}) {
  let lastError;
  for (let attempt = 1; attempt <= retries; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(url, {
        redirect: 'follow',
        ...options,
        signal: controller.signal,
      });
      clearTimeout(timer);

      // 429 e 5xx sono temporanei; 4xx (tranne 429) sono definitivi.
      if (response.status === 429 || response.status >= 500) {
        lastError = new Error(
          `HTTP ${response.status} ${response.statusText} — ${url}`
        );
      } else {
        return response;
      }
    } catch (err) {
      clearTimeout(timer);
      lastError = err;
    }

    if (attempt < retries) {
      const wait = 750 * 2 ** (attempt - 1);
      log(`  ↻ tentativo ${attempt}/${retries} fallito, riprovo tra ${wait} ms…`);
      await sleep(wait);
    }
  }
  throw new Error(
    `Impossibile scaricare ${url}: ${lastError?.message || 'errore di rete sconosciuto'}`
  );
}

async function responseToBuffer(response, label) {
  if (!response.ok) {
    throw new Error(`${label}: HTTP ${response.status} ${response.statusText}`);
  }
  return Buffer.from(await response.arrayBuffer());
}

function verifyIntegrity(buffer, integrity, label) {
  if (!integrity) return;
  const first = String(integrity).trim().split(/\s+/)[0];
  const dash = first.indexOf('-');
  if (dash <= 0) return;
  const algorithm = first.slice(0, dash);
  const expected = first.slice(dash + 1);
  const actual = createHash(algorithm).update(buffer).digest('base64');
  if (actual !== expected) {
    throw new Error(
      `${label}: checksum ${algorithm} non valido (atteso ${expected}, ottenuto ${actual})`
    );
  }
}

function sha256(buffer) {
  return createHash('sha256').update(buffer).digest('hex');
}

async function sha256File(path) {
  return await new Promise((resolvePromise, reject) => {
    const hash = createHash('sha256');
    createReadStream(path)
      .on('data', (chunk) => hash.update(chunk))
      .on('error', reject)
      .on('end', () => resolvePromise(hash.digest('hex')));
  });
}

function createProgress(label) {
  let lastDraw = 0;
  let lastPlain = 0;
  const start = Date.now();
  return {
    update(loaded, total) {
      const now = Date.now();
      const finished = total > 0 && loaded >= total;
      if (now - lastDraw < 200 && !finished) return;
      lastDraw = now;

      const elapsed = (now - start) / 1000;
      const speed = elapsed > 0 ? loaded / elapsed : 0;
      const pct = total > 0 ? Math.min(100, (loaded / total) * 100) : null;
      const eta = total > 0 && speed > 0 ? formatDuration((total - loaded) / speed) : '—';
      const parts = [
        `  ${label}: ${formatBytes(loaded)}${total > 0 ? ` / ${formatBytes(total)}` : ''}`,
        pct === null ? null : `${pct.toFixed(1)}%`,
        `${formatBytes(speed)}/s`,
        total > 0 ? `ETA ${eta}` : null,
      ].filter(Boolean);
      const line = parts.join('  •  ');

      if (process.stdout.isTTY) {
        process.stdout.write(`\r${line.padEnd(110).slice(0, 110)}`);
      } else if (finished || now - lastPlain > 5000) {
        // In CI/log non-TTY evitiamo migliaia di righe.
        log(line);
        lastPlain = now;
      }
    },
    done() {
      if (process.stdout.isTTY) process.stdout.write('\n');
    },
  };
}

/* ---------------------------------------------------------------------------
 * Estrazione tar.gz (solo file richiesti, senza dipendenze esterne)
 * ------------------------------------------------------------------------- */

function cString(buffer, offset, length) {
  const slice = buffer.subarray(offset, offset + length);
  const end = slice.indexOf(0);
  return slice.subarray(0, end === -1 ? slice.length : end).toString('utf8');
}

function parsePaxPath(buffer, offset, size) {
  const text = buffer.subarray(offset, offset + size).toString('utf8');
  let result = null;
  let cursor = 0;
  while (cursor < text.length) {
    const space = text.indexOf(' ', cursor);
    if (space === -1) break;
    const length = Number.parseInt(text.slice(cursor, space), 10);
    if (!Number.isFinite(length) || length <= 0) break;
    const record = text.slice(space + 1, cursor + length - 1);
    const eq = record.indexOf('=');
    if (eq > 0) {
      const key = record.slice(0, eq);
      const value = record.slice(eq + 1);
      if (key === 'path') result = value;
    }
    cursor += length;
  }
  return result;
}

/**
 * Estrae da un tarball gzip solo i path presenti in `wanted`.
 * Supporta header ustar, nomi GNU longname e record PAX "path".
 * @param {Buffer} tarball
 * @param {Set<string>} wanted
 * @returns {Map<string, Buffer>}
 */
function extractWantedFiles(tarball, wanted) {
  const tar = gunzipSync(tarball);
  const output = new Map();
  let offset = 0;
  let pendingPath = null;

  while (offset + 512 <= tar.length) {
    const header = tar.subarray(offset, offset + 512);
    if (header.every((byte) => byte === 0)) break;

    let name = cString(header, 0, 100);
    const prefix = cString(header, 345, 155);
    if (prefix) name = `${prefix}/${name}`;

    const sizeField = cString(header, 124, 12).trim();
    const size = sizeField ? Number.parseInt(sizeField, 8) : 0;
    if (!Number.isFinite(size) || size < 0) {
      throw new Error('Tarball non valido: dimensione file illeggibile.');
    }

    const type = header[156] === 0 ? '0' : String.fromCharCode(header[156]);
    const dataStart = offset + 512;

    if (type === 'L') {
      // GNU longname: il nome del prossimo file è nel body.
      pendingPath = cString(tar, dataStart, size);
    } else if (type === 'x') {
      pendingPath = parsePaxPath(tar, dataStart, size) || pendingPath;
    } else if (type === '0') {
      const finalName = pendingPath || name;
      if (wanted.has(finalName)) {
        output.set(finalName, Buffer.from(tar.subarray(dataStart, dataStart + size)));
      }
      pendingPath = null;
    } else {
      pendingPath = null;
    }

    offset = dataStart + Math.ceil(size / 512) * 512;
  }

  return output;
}

/* ---------------------------------------------------------------------------
 * Installazione generica di un pacchetto npm (tarball verificato)
 * ------------------------------------------------------------------------- */

/**
 * @typedef {object} PackageSpec
 * @property {string} label        nome leggibile ("Wllama", "WebLLM")
 * @property {string} packageName  nome npm ("@wllama/wllama")
 * @property {string} version
 * @property {string} dir          cartella di destinazione in vendor/
 * @property {Map<string,string>} wanted   path nel tarball -> path locale
 * @property {string[]} required   path locali obbligatori
 * @property {{tarball?:string, integrity?:string}} [env]
 * @property {(buffer: Buffer) => void} [validate]  controllo extra sul bundle
 */

async function packageLooksInstalled(spec) {
  for (const relative of spec.required) {
    if (!(await exists(join(spec.dir, relative)))) return false;
  }
  return true;
}

function describePackage(spec) {
  return `${spec.packageName} ${spec.version} → vendor/${basename(spec.dir)}`;
}

/** @param {PackageSpec} spec */
async function installPackage(spec) {
  logStep(`${spec.label} ${spec.version}`);

  if (CHECK_ONLY) {
    const installed = await packageLooksInstalled(spec);
    logSkip(
      installed
        ? `presente in ${spec.dir}`
        : `NON installato (atteso in ${spec.dir})`
    );
    return installed;
  }

  if (!FORCE && (await packageLooksInstalled(spec))) {
    logSkip(`già presente in ${spec.dir} (usa --force per riscaricarlo)`);
    return true;
  }

  const encodedName = spec.packageName.replace('/', '%2f');
  const metadataUrl = `${REGISTRY_URL}/${encodedName}/${encodeURIComponent(spec.version)}`;
  log('  • leggo i metadati npm…');
  const metaResponse = await request(metadataUrl, {}, { retries: 3 });
  const meta = JSON.parse(
    (
      await responseToBuffer(metaResponse, `Metadati npm ${spec.label}`)
    ).toString('utf8')
  );

  const tarballUrl = spec.env?.tarball || meta?.dist?.tarball;
  const integrity = spec.env?.integrity || meta?.dist?.integrity || '';
  if (!tarballUrl) {
    throw new Error(
      `Pacchetto ${spec.packageName}@${spec.version} non trovato nel registry npm.`
    );
  }

  log(`  • scarico ${tarballUrl}`);
  const tarballResponse = await request(tarballUrl, {}, {
    retries: 4,
    timeoutMs: 120_000,
  });
  const tarball = await responseToBuffer(tarballResponse, `Tarball ${spec.label}`);

  verifyIntegrity(tarball, integrity, `Tarball ${spec.label}`);
  logOk(`tarball verificato (${formatBytes(tarball.length)})`);

  const extracted = extractWantedFiles(tarball, new Set(spec.wanted.keys()));

  const missing = [];
  const files = [];
  for (const [source, relative] of spec.wanted) {
    const data = extracted.get(source);
    if (!data) {
      missing.push(`${relative} (atteso in ${source})`);
      continue;
    }
    files.push({ relative, data });
  }

  const missingRequired = spec.required.filter(
    (relative) => !files.some((file) => file.relative === relative)
  );
  if (missingRequired.length > 0) {
    throw new Error(
      `Il pacchetto ${spec.packageName}@${spec.version} non contiene ` +
        `${missingRequired.map((p) => `"${p}"`).join(', ')}. ` +
        `Contenuto mancante: ${missing.join(', ') || 'n/d'}`
    );
  }
  for (const entry of missing) logSkip(`facoltativo assente: ${entry}`);

  const bundle = files.find((file) => spec.required.includes(file.relative));
  if (bundle && spec.validate) spec.validate(bundle.data);

  const manifestFiles = [];
  for (const { relative, data } of files) {
    const destination = join(spec.dir, relative);
    await mkdir(dirname(destination), { recursive: true });
    // Scrittura via file temporaneo: niente bundle a metà se il processo muore.
    await atomicWrite(destination, data);
    manifestFiles.push({ path: relative, bytes: data.length, sha256: sha256(data) });
    logOk(`${relative} (${formatBytes(data.length)})`);
  }

  await atomicWrite(
    join(spec.dir, 'manifest.json'),
    JSON.stringify(
      {
        package: spec.packageName,
        version: spec.version,
        source: tarballUrl,
        integrity,
        installedAt: new Date().toISOString(),
        files: manifestFiles,
      },
      null,
      2
    ) + '\n'
  );

  logOk(`installato in ${spec.dir}`);
  return true;
}

/* ---------------------------------------------------------------------------
 * Specifiche dei pacchetti
 * ------------------------------------------------------------------------- */

/** @type {PackageSpec} */
const WLLAMA_SPEC = {
  label: 'Runtime Wllama',
  packageName: '@wllama/wllama',
  version: WLLAMA_VERSION,
  dir: WLLAMA_DIR,
  wanted: new Map([
    ['package/esm/index.js', 'esm/index.js'],
    ['package/esm/wasm/wllama.wasm', 'esm/wasm/wllama.wasm'],
    ['package/LICENSE', 'LICENSE'],
    ['package/LICENCE', 'LICENCE'],
    ['package/README.md', 'README.md'],
  ]),
  required: ['esm/index.js', 'esm/wasm/wllama.wasm'],
  env: {
    tarball: process.env.WLLAMA_TARBALL,
    integrity: process.env.WLLAMA_INTEGRITY,
  },
  validate(buffer) {
    if (buffer.length < 16 * 1024) {
      throw new Error('Wllama: esm/index.js sembra troncato.');
    }
  },
};

/**
 * Wllama 3.x usa JSPI/MEMORY64; sui browser che non li supportano (Safari,
 * Firefox senza flag) serve il pacchetto compat con worker + WASM dedicati.
 * Lo teniamo self-hosted così il runtime resta completamente offline.
 *
 * @type {PackageSpec}
 */
const WLLAMA_COMPAT_SPEC = {
  label: 'Runtime Wllama compat',
  packageName: '@wllama/wllama-compat',
  version: WLLAMA_VERSION,
  dir: WLLAMA_COMPAT_DIR,
  wanted: new Map([
    ['package/wasm/wllama.js', 'wasm/wllama.js'],
    ['package/wasm/wllama.wasm', 'wasm/wllama.wasm'],
    ['package/LICENSE', 'LICENSE'],
    ['package/LICENCE', 'LICENCE'],
    ['package/README.md', 'README.md'],
  ]),
  required: ['wasm/wllama.js', 'wasm/wllama.wasm'],
  env: {
    tarball: process.env.WLLAMA_COMPAT_TARBALL,
    integrity: process.env.WLLAMA_COMPAT_INTEGRITY,
  },
  validate(buffer) {
    if (buffer.length < 8 * 1024) {
      throw new Error('Wllama compat: wasm/wllama.js sembra troncato.');
    }
  },
};

/** @type {PackageSpec} */
const WEBLLM_SPEC = {
  label: 'Runtime WebLLM (backend GPU)',
  packageName: '@mlc-ai/web-llm',
  version: WEBLLM_VERSION,
  dir: WEBLLM_DIR,
  wanted: new Map([
    // `lib/index.js` è il bundle ESM self-contained (engine + worker inline).
    ['package/lib/index.js', 'lib/index.js'],
    ['package/LICENSE', 'LICENSE'],
    ['package/package.json', 'package.json'],
    ['package/README.md', 'README.md'],
  ]),
  required: ['lib/index.js'],
  env: {
    tarball: process.env.WEBLLM_TARBALL,
    integrity: process.env.WEBLLM_INTEGRITY,
  },
  validate(buffer) {
    const text = buffer.toString('utf8', 0, Math.min(buffer.length, 4_000_000));
    for (const symbol of ['CreateMLCEngine', 'MLCEngine']) {
      if (!text.includes(symbol)) {
        throw new Error(
          `WebLLM: lib/index.js non esporta "${symbol}" — bundle inatteso.`
        );
      }
    }
    if (buffer.length < 1024 * 1024) {
      throw new Error('WebLLM: lib/index.js sembra troncato.');
    }
  },
};

/* ---------------------------------------------------------------------------
 * Modello GGUF: download riprendibile, verifica dimensione/hash
 * ------------------------------------------------------------------------- */

async function getRemoteModelInfo() {
  try {
    const response = await request(MODEL_URL, { method: 'HEAD' }, {
      retries: 2,
      timeoutMs: 60_000,
    });
    if (!response.ok) return { size: null, etag: null };
    const length = Number(response.headers.get('content-length'));
    return {
      size: Number.isFinite(length) && length > 0 ? length : null,
      etag: response.headers.get('etag') || null,
    };
  } catch {
    return { size: null, etag: null };
  }
}

async function writeModelMeta(meta) {
  await atomicWrite(MODEL_META_PATH, JSON.stringify(meta, null, 2) + '\n');
}

async function finalModelIsValid(remote) {
  const size = await fileSize(MODEL_PATH);
  if (size === null) return false;
  if (remote.size !== null && size !== remote.size) return false;

  if (MODEL_SHA256) {
    const actual = await sha256File(MODEL_PATH);
    if (actual !== MODEL_SHA256) {
      log(`  ! checksum del modello non valido: ${actual}`);
      return false;
    }
  }
  return true;
}

async function downloadModel() {
  logStep(`Modello GGUF: ${MODEL_FILE}`);

  const remote = await getRemoteModelInfo();
  if (remote.size !== null) {
    log(`  • dimensione remota: ${formatBytes(remote.size)}`);
  }

  if (CHECK_ONLY) {
    const size = await fileSize(MODEL_PATH);
    logSkip(
      size === null
        ? `NON presente (atteso in ${MODEL_PATH})`
        : `presente in ${MODEL_PATH} (${formatBytes(size)})`
    );
    return size !== null;
  }

  if (!FORCE && (await finalModelIsValid(remote))) {
    logSkip(`già presente in ${MODEL_PATH}`);
    return true;
  }

  await mkdir(MODEL_DIR, { recursive: true });

  if (FORCE) {
    await rm(MODEL_PATH, { force: true });
    await rm(MODEL_PART_PATH, { force: true });
    await rm(MODEL_META_PATH, { force: true });
  }

  // Se esiste un file finale incompleto e non c'è un .part, lo riusiamo come
  // base per il resume.
  const finalSize = await fileSize(MODEL_PATH);
  if (finalSize !== null && !(await exists(MODEL_PART_PATH))) {
    if (remote.size !== null && finalSize >= remote.size) {
      await rm(MODEL_PATH, { force: true });
    } else {
      await rename(MODEL_PATH, MODEL_PART_PATH);
    }
  }

  const progress = createProgress('modello');
  const maxAttempts = 5;
  let lastError;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    let start = (await fileSize(MODEL_PART_PATH)) || 0;
    if (remote.size !== null && start > remote.size) {
      await rm(MODEL_PART_PATH, { force: true });
      start = 0;
    }
    if (remote.size !== null && start === remote.size) break;

    const headers = {};
    if (start > 0) headers.Range = `bytes=${start}-`;

    log(
      start > 0
        ? `  ↻ riprendo da ${formatBytes(start)} (tentativo ${attempt}/${maxAttempts})`
        : `  • avvio download (tentativo ${attempt}/${maxAttempts})`
    );

    let response;
    try {
      response = await request(MODEL_URL, { headers }, {
        retries: 1,
        timeoutMs: 120_000,
      });
    } catch (err) {
      lastError = err;
      if (attempt < maxAttempts) {
        await sleep(1000 * attempt);
        continue;
      }
      throw err;
    }

    if (response.status === 416) {
      // Il file locale è già completo secondo il server.
      const size = (await fileSize(MODEL_PART_PATH)) || 0;
      if (remote.size === null || size === remote.size) break;
      await rm(MODEL_PART_PATH, { force: true });
      continue;
    }

    if (!response.ok) {
      throw new Error(
        `Download modello fallito: HTTP ${response.status} ${response.statusText}`
      );
    }

    const contentRange = response.headers.get('content-range');
    if (start > 0 && response.status === 206 && contentRange) {
      const match = /^bytes\s+(\d+)-/i.exec(contentRange);
      if (!match || Number(match[1]) !== start) {
        log('  ! il server ha restituito un range inatteso, riparto da zero');
        await rm(MODEL_PART_PATH, { force: true });
        continue;
      }
    } else if (start > 0 && response.status === 200) {
      // Range ignorato: il body ricomincia dal byte 0.
      log('  • il server non supporta Range, riparto da zero');
      start = 0;
      await rm(MODEL_PART_PATH, { force: true });
    }

    const totalFromResponse = Number(response.headers.get('content-length'));
    const total =
      start > 0 || remote.size === null
        ? start + (Number.isFinite(totalFromResponse) ? totalFromResponse : 0)
        : remote.size;

    const handle = await open(MODEL_PART_PATH, start > 0 ? 'r+' : 'w');
    let writeOffset = start;
    let loaded = start;

    try {
      if (start > 0) await handle.truncate(start);

      const body = Readable.fromWeb(response.body);
      for await (const chunk of body) {
        const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
        await handle.write(buffer, 0, buffer.length, writeOffset);
        writeOffset += buffer.length;
        loaded += buffer.length;
        progress.update(loaded, total);
      }

      await handle.sync();
    } catch (err) {
      lastError = err;
      await handle.close().catch(() => {});
      if (attempt < maxAttempts) {
        log(`  ↻ download interrotto (${err.message}), riprovo…`);
        await sleep(1000 * attempt);
        continue;
      }
      throw err;
    } finally {
      await handle.close().catch(() => {});
      progress.done();
    }

    const downloadedSize = await fileSize(MODEL_PART_PATH);
    if (remote.size !== null && downloadedSize !== remote.size) {
      await rm(MODEL_PART_PATH, { force: true });
      lastError = new Error(
        `Dimensione non valida: ${formatBytes(downloadedSize)} invece di ${formatBytes(remote.size)}.`
      );
      if (attempt < maxAttempts) continue;
      throw lastError;
    }

    break;
  }

  const partSize = await fileSize(MODEL_PART_PATH);
  if (partSize === null) {
    throw lastError || new Error('Download del modello non completato.');
  }
  if (remote.size !== null && partSize !== remote.size) {
    throw new Error(
      `Download incompleto: ${formatBytes(partSize)} / ${formatBytes(remote.size)}. Rilancia il comando per riprendere.`
    );
  }

  if (MODEL_SHA256) {
    log('  • verifico checksum SHA-256…');
    const actual = await sha256File(MODEL_PART_PATH);
    if (actual !== MODEL_SHA256) {
      await rm(MODEL_PART_PATH, { force: true });
      throw new Error(
        `Checksum SHA-256 non valido: atteso ${MODEL_SHA256}, ottenuto ${actual}.`
      );
    }
    logOk('checksum SHA-256 verificato');
  }

  await rename(MODEL_PART_PATH, MODEL_PATH);
  await writeModelMeta({
    url: MODEL_URL,
    file: MODEL_FILE,
    bytes: partSize,
    etag: remote.etag,
    sha256: MODEL_SHA256 || null,
    downloadedAt: new Date().toISOString(),
  });

  logOk(`modello installato in ${MODEL_PATH} (${formatBytes(partSize)})`);
  return true;
}

/* ---------------------------------------------------------------------------
 * Main
 * ------------------------------------------------------------------------- */

async function main() {
  log('');
  log('╭──────────────────────────────────────────────────────────────╮');
  log('│  Download dipendenze runtime — modalità offline/self-hosted  │');
  log('╰──────────────────────────────────────────────────────────────╯');
  log(`  root: ${ROOT}`);
  if (CHECK_ONLY) log('  modalità: solo verifica (--check)');

  const failures = [];

  if (SKIP_WLLAMA) {
    logStep('Runtime Wllama: saltato (--skip-wllama / --skip-runtime)');
  } else {
    try {
      await installPackage(WLLAMA_SPEC);
      await installPackage(WLLAMA_COMPAT_SPEC);
    } catch (err) {
      failures.push(`Wllama: ${err.message}`);
      log(`  ✗ ${err.message}`);
    }
  }

  if (SKIP_WEBLLM) {
    logStep('Runtime WebLLM: saltato (--skip-webllm / --skip-runtime)');
  } else {
    try {
      await installPackage(WEBLLM_SPEC);
    } catch (err) {
      failures.push(`WebLLM: ${err.message}`);
      log(`  ✗ ${err.message}`);
    }
  }

  if (SKIP_MODEL) {
    logStep('Modello GGUF: saltato (--skip-model)');
  } else {
    try {
      await downloadModel();
    } catch (err) {
      failures.push(`Modello: ${err.message}`);
      log(`  ✗ ${err.message}`);
    }
  }

  if (CHECK_ONLY) {
    log('');
    return;
  }

  if (failures.length > 0) {
    log('');
    log('❌ Installazione incompleta:');
    for (const failure of failures) log(`   • ${failure}`);
    log('  Rilancia il comando: i download già completati vengono saltati.');
    log('');
    process.exitCode = 1;
    return;
  }

  if (SKIP_WLLAMA) log('\n⚠ Runtime Wllama saltato: il backend CPU non sarà disponibile.');
  if (SKIP_WEBLLM) log('\n⚠ Runtime WebLLM saltato: il backend GPU userà il CDN.');
  if (SKIP_MODEL) {
    log('\n⚠ Modello GGUF saltato su richiesta (--skip-model).');
    log('  Esegui il comando senza --skip-model prima di aprire index.html.');
  }

  if (SKIP_WLLAMA || SKIP_WEBLLM || SKIP_MODEL) {
    log('');
    return;
  }

  log('\n✅ Tutto pronto: avvia il server con "bun run serve" (o "node server.js").');
  log('   L’applicazione ora carica solo file locali: nessuna CDN o API esterna.');
  log('   Backend GPU: vendor/web-llm/lib/index.js (i pesi MLC li scarica il browser).');
  log('');
}

main().catch((err) => {
  console.error(`\n❌ ${err?.message || err}\n`);
  if (process.env.DEBUG) console.error(err);
  process.exitCode = 1;
});
