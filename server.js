#!/usr/bin/env node
/**
 * ============================================================================
 * server.js — server statico minimale per GGUFEngine
 * ============================================================================
 * Serve la cartella del progetto con le intestazioni necessarie a WebGPU e ai
 * thread WASM:
 *
 *   Cross-Origin-Opener-Policy:   same-origin
 *   Cross-Origin-Embedder-Policy: require-corp
 *
 * Supporta richieste Range (ripresa del download dei modelli) e non richiede
 * dipendenze: solo moduli Node.
 *
 * Uso:
 *   node server.js            # http://127.0.0.1:8080
 *   PORT=3000 HOST=0.0.0.0 node server.js
 * ============================================================================
 */
import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import { createServer } from 'node:http';
import { extname, join, normalize, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(new URL('.', import.meta.url)));
const PORT = Number(process.env.PORT) || 8080;
const HOST = process.env.HOST || '127.0.0.1';

const MIME = new Map(Object.entries({
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
  '.gguf': 'application/octet-stream',
  '.bin': 'application/octet-stream',
  '.txt': 'text/plain; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.ico': 'image/x-icon',
}));

function commonHeaders(extra = {}) {
  return {
    // Necessari per SharedArrayBuffer (WASM multi-thread) e per un contesto
    // sicuro stabile: senza questi header i thread WASM non partono.
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
    'Cross-Origin-Resource-Policy': 'same-origin',
    'X-Content-Type-Options': 'nosniff',
    ...extra,
  };
}

function send(res, status, body, headers = {}) {
  const payload = Buffer.from(body);
  res.writeHead(status, commonHeaders({
    'Content-Type': 'text/plain; charset=utf-8',
    'Content-Length': payload.length,
    ...headers,
  }));
  res.end(payload);
}

function resolvePath(urlPath) {
  let decoded;
  try {
    decoded = decodeURIComponent(urlPath.split('?')[0].split('#')[0]);
  } catch {
    return null;
  }
  const relative = normalize(decoded).replace(/^([/\\])+/, '');
  const target = resolve(join(ROOT, relative || 'index.html'));
  if (target !== ROOT && !target.startsWith(ROOT + sep)) return null;
  return target;
}

const server = createServer(async (req, res) => {
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    send(res, 405, 'Metodo non consentito', { Allow: 'GET, HEAD' });
    return;
  }

  let target = resolvePath(req.url || '/');
  if (!target) {
    send(res, 400, 'Percorso non valido');
    return;
  }

  let info;
  try {
    info = await stat(target);
    if (info.isDirectory()) {
      target = join(target, 'index.html');
      info = await stat(target);
    }
  } catch {
    send(res, 404, 'File non trovato');
    return;
  }

  const type = MIME.get(extname(target).toLowerCase()) || 'application/octet-stream';
  const isHtml = type.startsWith('text/html');
  const headers = commonHeaders({
    'Content-Type': type,
    'Accept-Ranges': 'bytes',
    'Cache-Control': isHtml ? 'no-store' : 'public, max-age=3600',
  });

  const headOnly = req.method === 'HEAD';
  const range = req.headers.range;

  if (range) {
    const match = /^bytes=(\d*)-(\d*)$/i.exec(String(range).trim());
    if (!match) {
      send(res, 416, 'Range non valido', { 'Content-Range': `bytes */${info.size}` });
      return;
    }
    let start = match[1] === '' ? null : Number(match[1]);
    let end = match[2] === '' ? null : Number(match[2]);
    if (start === null && end !== null) {
      // Suffisso: ultimi N byte.
      start = Math.max(0, info.size - end);
      end = info.size - 1;
    } else {
      start = start ?? 0;
      end = end === null ? info.size - 1 : Math.min(end, info.size - 1);
    }
    if (!Number.isFinite(start) || !Number.isFinite(end) || start > end || start >= info.size) {
      send(res, 416, 'Range non soddisfacibile', { 'Content-Range': `bytes */${info.size}` });
      return;
    }

    const length = end - start + 1;
    res.writeHead(206, {
      ...headers,
      'Content-Range': `bytes ${start}-${end}/${info.size}`,
      'Content-Length': length,
    });
    if (headOnly) {
      res.end();
      return;
    }
    createReadStream(target, { start, end }).on('error', () => res.destroy()).pipe(res);
    return;
  }

  res.writeHead(200, { ...headers, 'Content-Length': info.size });
  if (headOnly) {
    res.end();
    return;
  }
  createReadStream(target).on('error', () => res.destroy()).pipe(res);
});

server.listen(PORT, HOST, () => {
  const url = `http://${HOST}:${PORT}/`;
  console.log('');
  console.log('  GGUFEngine');
  console.log(`  ➜  ${url}`);
  console.log(`  root: ${ROOT}`);
  console.log('  COOP/COEP attivi: WebGPU e thread WASM disponibili.');
  console.log('  Ferma il server con Ctrl+C.');
  console.log('');
});
