#!/usr/bin/env bash
# =============================================================================
# download-deps.sh — scarica tutte le dipendenze runtime (versione Bash)
# =============================================================================
# Equivalente di scripts/download-deps.js, senza Node. Dopo l'esecuzione
# l'applicazione non contatta più CDN o Hugging Face:
#
#   vendor/wllama/esm/index.js                    runtime @wllama/wllama
#   vendor/wllama/esm/wasm/wllama.wasm            WASM (v3, JSPI/MEMORY64)
#   vendor/wllama-compat/wasm/wllama.js           compat worker (Safari/ecc.)
#   vendor/wllama-compat/wasm/wllama.wasm         compat WASM
#   vendor/web-llm/lib/index.js                   runtime WebLLM (backend GPU)
#   models/MNN-Pocket-0.3B-Instruct-Q4_1.gguf     modello GGUF
#
# Uso:
#   bash scripts/download-deps.sh
#   bash scripts/download-deps.sh --force --skip-model
#
# Opzioni:
#   --force                 riscarica anche ciò che è già presente
#   --skip-wllama           non scarica Wllama (runtime CPU)
#   --skip-webllm           non scarica WebLLM (runtime GPU)
#   --skip-runtime          non scarica nessuno dei due runtime
#   --skip-model            non scarica il modello GGUF
#   --check                 mostra solo lo stato dei file, senza scaricare
#   --with-webllm           compatibilità: WebLLM è già il default (no-op)
#   --wllama-version=X.Y.Z
#   --webllm-version=X.Y.Z
#   --model-url=URL
#   --model-file=NOME.gguf
#
# Variabili d'ambiente equivalenti:
#   WLLAMA_VERSION, WLLAMA_COMPAT_VERSION,
#   WLLAMA_TARBALL, WLLAMA_INTEGRITY,
#   WLLAMA_COMPAT_TARBALL, WLLAMA_COMPAT_INTEGRITY,
#   WEBLLM_VERSION, WEBLLM_TARBALL, WEBLLM_INTEGRITY,
#   MODEL_URL, MODEL_FILE, MODEL_SHA256,
#   FORCE=1, SKIP_WLLAMA=1, SKIP_WEBLLM=1, SKIP_MODEL=1, CHECK=1
#
# Il download del modello è riprendibile (HTTP Range): se viene interrotto,
# rilanciando il comando riparte dal byte successivo.
# =============================================================================
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_ROOT="$ROOT/vendor"
WLLAMA_DIR="$VENDOR_ROOT/wllama"
WLLAMA_COMPAT_DIR="$VENDOR_ROOT/wllama-compat"
WEBLLM_DIR="$VENDOR_ROOT/web-llm"
MODEL_DIR="$ROOT/models"
REGISTRY_URL="https://registry.npmjs.org"

DEFAULT_MODEL_URL='https://huggingface.co/taobao-mnn/MNN-Pocket-0.3B-GGUF/resolve/main/MNN-Pocket-0.3B-Instruct-Q4_1.gguf'

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'; C_DIM=$'\033[2m'; C_OK=$'\033[32m'; C_ERR=$'\033[31m'; C_HEAD=$'\033[1m'
else
  C_RESET=''; C_DIM=''; C_OK=''; C_ERR=''; C_HEAD=''
fi

log()      { printf '%s\n' "$*"; }
log_step() { printf '\n%s▶ %s%s\n' "$C_HEAD" "$*" "$C_RESET"; }
log_ok()   { printf '  %s✓%s %s\n' "$C_OK" "$C_RESET" "$*"; }
log_skip() { printf '  %s•%s %s\n' "$C_DIM" "$C_RESET" "$*"; }
log_warn() { printf '  ! %s\n' "$*"; }
log_err()  { printf '  %s✗%s %s\n' "$C_ERR" "$C_RESET" "$*" >&2; }
die()      { printf '\n%s❌ %s%s\n\n' "$C_ERR" "$*" "$C_RESET" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Flag / ambiente
# ---------------------------------------------------------------------------
FORCE="${FORCE:-0}"
SKIP_WLLAMA="${SKIP_WLLAMA:-0}"
SKIP_WEBLLM="${SKIP_WEBLLM:-0}"
SKIP_MODEL="${SKIP_MODEL:-0}"
CHECK_ONLY="${CHECK:-0}"
WITH_WEBLLM_LEGACY=0

norm_bool() { # 1/true/yes/on -> 1 ; 0/false/no/off -> 0 ; altrimenti vuoto
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on)  printf '1' ;;
    0|false|no|off) printf '0' ;;
    *)              printf '' ;;
  esac
}

set_flag() { # nome variabile, valore testuale
  local value; value="$(norm_bool "$2")"
  [[ -n "$value" ]] || die "Valore booleano non valido per --$1: '$2'"
  printf -v "$3" '%s' "$value"
}

WLLAMA_VERSION="${WLLAMA_VERSION:-3.13.0}"
WLLAMA_COMPAT_VERSION="${WLLAMA_COMPAT_VERSION:-$WLLAMA_VERSION}"
WEBLLM_VERSION="${WEBLLM_VERSION:-0.2.89}"
MODEL_URL="${MODEL_URL:-$DEFAULT_MODEL_URL}"

while (($# > 0)); do
  arg="$1"; shift
  case "$arg" in
    --force)             FORCE=1 ;;
    --force=*)           set_flag force "${arg#*=}" FORCE ;;
    --skip-runtime)      SKIP_WLLAMA=1; SKIP_WEBLLM=1 ;;
    --skip-runtime=*)    set_flag skip-runtime "${arg#*=}" v; SKIP_WLLAMA="$v"; SKIP_WEBLLM="$v" ;;
    --skip-wllama)       SKIP_WLLAMA=1 ;;
    --skip-wllama=*)     set_flag skip-wllama "${arg#*=}" SKIP_WLLAMA ;;
    --skip-webllm)       SKIP_WEBLLM=1 ;;
    --skip-webllm=*)     set_flag skip-webllm "${arg#*=}" SKIP_WEBLLM ;;
    --skip-model)        SKIP_MODEL=1 ;;
    --skip-model=*)      set_flag skip-model "${arg#*=}" SKIP_MODEL ;;
    --check)             CHECK_ONLY=1 ;;
    --check=*)           set_flag check "${arg#*=}" CHECK_ONLY ;;
    --with-webllm)       WITH_WEBLLM_LEGACY=1 ;;
    --wllama-version=*)  WLLAMA_VERSION="${arg#*=}" ;;
    --webllm-version=*)  WEBLLM_VERSION="${arg#*=}" ;;
    --model-url=*)       MODEL_URL="${arg#*=}" ;;
    --model-file=*)      MODEL_FILE="${arg#*=}" ;;
    -h|--help)           awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)                   die "Opzione sconosciuta: $arg (usa --help)" ;;
  esac
done

(( WITH_WEBLLM_LEGACY )) && (( SKIP_WLLAMA && SKIP_WEBLLM )) && \
  log_warn '--with-webllm ignorato: --skip-runtime ha la precedenza.'

MODEL_FILE="${MODEL_FILE:-$(basename "${MODEL_URL%%\?*}")}"
MODEL_FILE="${MODEL_FILE:-model.gguf}"
MODEL_PATH="$MODEL_DIR/$MODEL_FILE"
MODEL_PART_PATH="$MODEL_PATH.part"
MODEL_META_PATH="$MODEL_PATH.json"
MODEL_SHA256="$(printf '%s' "${MODEL_SHA256:-}" | tr '[:upper:]' '[:lower:]')"

# ---------------------------------------------------------------------------
# Requisiti
# ---------------------------------------------------------------------------
need() {
  command -v "$1" >/dev/null 2>&1 || die "Comando richiesto non trovato: $1"
}
need curl
need tar

# La barra di avanzamento di curl scrive su stderr: mostrala solo su terminale.
if [[ -t 2 ]]; then
  CURL_QUIET=(--progress-bar)
else
  CURL_QUIET=(--silent --show-error)
fi

sha256_of_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}'
  elif command -v openssl >/dev/null 2>&1; then openssl dgst -sha256 "$1" | awk '{print $NF}'
  else return 1; fi
}

file_size() {
  [[ -f "$1" ]] && wc -c < "$1" | tr -d ' ' || printf ''
}

format_bytes() {
  awk -v b="${1:-0}" 'BEGIN{
    split("B KB MB GB TB", u, " ");
    i=1;
    while (b>=1024 && i<5) { b/=1024; i++ }
    if (i==1) printf "%d %s", b, u[i]; else printf "%.1f %s", b, u[i];
  }'
}

TMP_DIRS=()
cleanup() {
  local dir
  for dir in "${TMP_DIRS[@]:-}"; do
    [[ -n "$dir" ]] && rm -rf "$dir"
  done
  return 0
}
trap cleanup EXIT

new_tmpdir() {
  local dir; dir="$(mktemp -d "${TMPDIR:-/tmp}/gguf-deps.XXXXXX")"
  TMP_DIRS+=("$dir")
  printf '%s' "$dir"
}

# ---------------------------------------------------------------------------
# Download con retry
# ---------------------------------------------------------------------------
fetch() { # url dest — il dest può essere "" per HEAD
  local url="$1" dest="$2" attempt
  if [[ -z "$dest" ]]; then
    curl -fsSL --retry 3 --retry-delay 1 --connect-timeout 30 \
      --max-time 60 --head "$url" 2>/dev/null
    return 0
  fi
  for attempt in 1 2 3 4; do
    if curl -fL --retry 3 --retry-delay 1 --connect-timeout 30 \
         "${CURL_QUIET[@]}" -o "$dest" "$url"; then
      return 0
    fi
    log_warn "tentativo $attempt/4 fallito, riprovo…"
    sleep $((attempt))
  done
  return 1
}

fetch_json() { # url -> stdout
  local url="$1"
  curl -fsSL --retry 3 --retry-delay 1 --connect-timeout 30 --max-time 120 "$url"
}

json_field() { # chiave — legge JSON da stdin (estrazione semplice, senza jq)
  local key="$1"
  tr -d '\n' | grep -o "\"$key\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" | head -n1 | \
    sed 's/.*:[[:space:]]*"//; s/"$//'
}

verify_integrity() { # file, integrity(sri), label
  local file="$1" integrity="$2" label="$3"
  [[ -n "$integrity" ]] || return 0
  local token algo expected actual
  token="${integrity%% *}"
  algo="${token%%-*}"
  expected="${token#*-}"
  [[ -n "$algo" && "$expected" != "$token" ]] || return 0
  if ! command -v openssl >/dev/null 2>&1; then
    log_warn "$label: openssl assente, checksum non verificato"
    return 0
  fi
  actual="$(openssl dgst "-$algo" -binary "$file" | openssl base64 -A)"
  if [[ "$actual" != "$expected" ]]; then
    die "$label: checksum $algo non valido (atteso $expected, ottenuto $actual)"
  fi
  log_ok "checksum $algo verificato"
}

# ---------------------------------------------------------------------------
# Installazione di un pacchetto npm (tarball verificato)
# ---------------------------------------------------------------------------
# install_npm_package <label> <nome-pkg> <versione> <dest-dir> <required|-> \
#     <tarball-env> <integrity-env> <src:dest>...
install_npm_package() {
  local label="$1" pkg="$2" version="$3" dest="$4" required_csv="$5"
  local tarball_env="$6" integrity_env="$7"; shift 7
  local pairs=("$@")
  local relative required

  log_step "$label $version"

  if (( CHECK_ONLY )); then
    local missing=0
    IFS=',' read -r -a _req <<< "$required_csv"
    for relative in "${_req[@]}"; do
      [[ -n "$relative" && ! -f "$dest/$relative" ]] && missing=1
    done
    if (( missing )); then
      log_skip "NON installato (atteso in $dest)"
      return 1
    fi
    log_skip "presente in $dest"
    return 0
  fi

  if (( ! FORCE )); then
    local installed=1
    IFS=',' read -r -a _req <<< "$required_csv"
    for relative in "${_req[@]}"; do
      [[ -n "$relative" && ! -f "$dest/$relative" ]] && installed=0
    done
    if (( installed )); then
      log_skip "già presente in $dest (usa --force per riscaricarlo)"
      return 0
    fi
  fi

  local encoded meta tarball_url integrity
  encoded="$(printf '%s' "$pkg" | sed 's|/|%2f|')"
  log '  • leggo i metadati npm…'
  meta="$(fetch_json "$REGISTRY_URL/$encoded/$version")" || \
    die "$label: metadati npm non raggiungibili ($REGISTRY_URL/$encoded/$version)"

  tarball_url="${!tarball_env:-}"
  [[ -n "$tarball_url" ]] || tarball_url="$(printf '%s' "$meta" | json_field tarball)"
  integrity="${!integrity_env:-}"
  [[ -n "$integrity" ]] || integrity="$(printf '%s' "$meta" | json_field integrity)"
  [[ -n "$tarball_url" ]] || die "Pacchetto $pkg@$version non trovato nel registry npm."

  local tmp tgz tar_list
  tmp="$(new_tmpdir)"
  tgz="$tmp/package.tgz"

  log "  • scarico $tarball_url"
  fetch "$tarball_url" "$tgz" || die "$label: download del tarball fallito"
  verify_integrity "$tgz" "$integrity" "Tarball $label"
  log_ok "tarball scaricato ($(format_bytes "$(file_size "$tgz")"))"

  tar_list="$tmp/list.txt"
  tar -tzf "$tgz" > "$tar_list" || die "$label: tarball non leggibile"

  # Estrai solo i file richiesti e presenti.
  local -a members=()
  local pair src
  for pair in "${pairs[@]}"; do
    src="${pair%%:*}"
    grep -Fxq "$src" "$tar_list" && members+=("$src")
  done
  if ((${#members[@]} > 0)); then
    local raw="$tmp/raw"
    mkdir -p "$raw"
    tar -xzf "$tgz" -C "$raw" --no-same-owner --no-same-permissions "${members[@]}"
  fi

  # Controlla i file obbligatori.
  local -a reqs=()
  IFS=',' read -r -a reqs <<< "$required_csv"
  local missing_required=()
  for relative in "${reqs[@]}"; do
    [[ -n "$relative" ]] || continue
    local found=0
    for pair in "${pairs[@]}"; do
      [[ "${pair##*:}" == "$relative" && -f "$tmp/raw/${pair%%:*}" ]] && found=1 && break
    done
    (( found )) || missing_required+=("$relative")
  done
  if ((${#missing_required[@]} > 0)); then
    die "$label: il pacchetto $pkg@$version non contiene ${missing_required[*]}"
  fi

  mkdir -p "$dest"
  local -a manifest_entries=()
  local target size hash tmp_target
  for pair in "${pairs[@]}"; do
    src="${pair%%:*}"
    target="${pair#*:}"
    if [[ ! -f "$tmp/raw/$src" ]]; then
      log_skip "facoltativo assente: $target"
      continue
    fi
    mkdir -p "$(dirname "$dest/$target")"
    tmp_target="$dest/$target.part-$$"
    cp -f "$tmp/raw/$src" "$tmp_target"
    mv -f "$tmp_target" "$dest/$target"
    size="$(file_size "$dest/$target")"
    hash="$(sha256_of_file "$dest/$target" || printf '')"
    log_ok "$target ($(format_bytes "$size"))"
    manifest_entries+=("{\"path\":\"$target\",\"bytes\":$size,\"sha256\":\"$hash\"}")
  done

  # Sanity check specifico del bundle.
  local bundle="$dest/${reqs[0]}"
  case "$pkg" in
    '@mlc-ai/web-llm')
      if ! grep -qa 'MLCEngine' "$bundle" 2>/dev/null; then
        die 'WebLLM: lib/index.js non esporta MLCEngine — bundle inatteso.'
      fi
      (( $(file_size "$bundle") > 1048576 )) || die 'WebLLM: lib/index.js sembra troncato.'
      ;;
    '@wllama/wllama')
      (( $(file_size "$bundle") > 16384 )) || die 'Wllama: esm/index.js sembra troncato.'
      ;;
    '@wllama/wllama-compat')
      (( $(file_size "$bundle") > 8192 )) || die 'Wllama compat: wasm/wllama.js sembra troncato.'
      ;;
  esac

  local files_json
  files_json="$(IFS=,; printf '%s' "${manifest_entries[*]:-}")"
  printf '{\n  "package": "%s",\n  "version": "%s",\n  "source": "%s",\n  "integrity": "%s",\n  "installedAt": "%s",\n  "files": [%s]\n}\n' \
    "$pkg" "$version" "$tarball_url" "$integrity" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$files_json" > "$dest/manifest.json.tmp-$$"
  mv -f "$dest/manifest.json.tmp-$$" "$dest/manifest.json"

  log_ok "installato in $dest"
  return 0
}

# ---------------------------------------------------------------------------
# Modello GGUF
# ---------------------------------------------------------------------------
remote_size() {
  local len
  len="$(fetch "$MODEL_URL" "" | tr -d '\r' | \
        awk 'BEGIN{IGNORECASE=1} /^content-length:/{v=$2} END{print v}')" || true
  [[ "$len" =~ ^[0-9]+$ ]] && printf '%s' "$len" || printf ''
}

download_model() {
  log_step "Modello GGUF: $MODEL_FILE"

  local expected; expected="$(remote_size)"
  [[ -n "$expected" ]] && log "  • dimensione remota: $(format_bytes "$expected")"

  if (( CHECK_ONLY )); then
    if [[ -f "$MODEL_PATH" ]]; then
      log_skip "presente in $MODEL_PATH ($(format_bytes "$(file_size "$MODEL_PATH")"))"
      return 0
    fi
    log_skip "NON presente (atteso in $MODEL_PATH)"
    return 1
  fi

  if (( ! FORCE )) && [[ -f "$MODEL_PATH" ]]; then
    if [[ -z "$expected" || "$(file_size "$MODEL_PATH")" == "$expected" ]]; then
      if [[ -z "$MODEL_SHA256" ]] || \
         [[ "$(sha256_of_file "$MODEL_PATH" || printf '')" == "$MODEL_SHA256" ]]; then
        log_skip "già presente in $MODEL_PATH"
        return 0
      fi
      log_warn 'checksum del modello non valido, riscarico'
    fi
  fi

  mkdir -p "$MODEL_DIR"

  if (( FORCE )); then
    rm -f "$MODEL_PATH" "$MODEL_PART_PATH" "$MODEL_META_PATH"
  elif [[ -f "$MODEL_PATH" && ! -f "$MODEL_PART_PATH" ]]; then
    # File finale incompleto: usalo come base per il resume.
    if [[ -n "$expected" && "$(file_size "$MODEL_PATH")" -ge "$expected" ]]; then
      rm -f "$MODEL_PATH"
    else
      mv -f "$MODEL_PATH" "$MODEL_PART_PATH"
    fi
  fi

  local attempt start rc=0
  for attempt in 1 2 3 4 5; do
    start="$(file_size "$MODEL_PART_PATH")"; start="${start:-0}"
    if [[ -n "$expected" && "$start" -ge "$expected" ]]; then
      break
    fi

    if (( start > 0 )); then
      log "  ↻ riprendo da $(format_bytes "$start") (tentativo $attempt/5)"
    else
      log "  • avvio download (tentativo $attempt/5)"
    fi

    # -C - gestisce il Range; se il server lo ignora curl riparte da zero.
    if curl -fL --retry 3 --retry-delay 2 --connect-timeout 30 \
         --speed-time 120 --speed-limit 1024 \
         -C - -o "$MODEL_PART_PATH" "${CURL_QUIET[@]}" "$MODEL_URL"; then
      rc=0
    else
      rc=$?
      log_warn "download interrotto (curl exit $rc), riprovo…"
      # 33 = il server non supporta il resume: riparti da zero.
      if [[ "$rc" -eq 33 ]]; then rm -f "$MODEL_PART_PATH"; fi
      sleep $((attempt))
      continue
    fi

    if [[ -n "$expected" && "$(file_size "$MODEL_PART_PATH")" != "$expected" ]]; then
      log_warn 'dimensione non valida, riparto…'
      rm -f "$MODEL_PART_PATH"
      continue
    fi
    break
  done

  [[ -f "$MODEL_PART_PATH" ]] || die 'Download del modello non completato.'
  local size; size="$(file_size "$MODEL_PART_PATH")"
  if [[ -n "$expected" && "$size" != "$expected" ]]; then
    log_err "Download incompleto: $(format_bytes "$size") / $(format_bytes "$expected")"
    log '  Rilancia il comando per riprendere dal byte successivo.'
    return 1
  fi

  if [[ -n "$MODEL_SHA256" ]]; then
    log '  • verifico checksum SHA-256…'
    local actual; actual="$(sha256_of_file "$MODEL_PART_PATH" || printf '')"
    if [[ "$actual" != "$MODEL_SHA256" ]]; then
      rm -f "$MODEL_PART_PATH"
      die "Checksum SHA-256 non valido: atteso $MODEL_SHA256, ottenuto $actual"
    fi
    log_ok 'checksum SHA-256 verificato'
  fi

  mv -f "$MODEL_PART_PATH" "$MODEL_PATH"
  printf '{\n  "url": "%s",\n  "file": "%s",\n  "bytes": %s,\n  "sha256": %s,\n  "downloadedAt": "%s"\n}\n' \
    "$MODEL_URL" "$MODEL_FILE" "$size" \
    "$([[ -n "$MODEL_SHA256" ]] && printf '"%s"' "$MODEL_SHA256" || printf 'null')" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$MODEL_META_PATH.tmp-$$"
  mv -f "$MODEL_META_PATH.tmp-$$" "$MODEL_META_PATH"

  log_ok "modello installato in $MODEL_PATH ($(format_bytes "$size"))"
  return 0
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
  log ''
  log '╭──────────────────────────────────────────────────────────────╮'
  log '│  Download dipendenze runtime — modalità offline/self-hosted  │'
  log '╰──────────────────────────────────────────────────────────────╯'
  log "  root: $ROOT"
  (( CHECK_ONLY )) && log '  modalità: solo verifica (--check)'

  local failures=0

  if (( SKIP_WLLAMA )); then
    log_step 'Runtime Wllama: saltato (--skip-wllama / --skip-runtime)'
  else
    install_npm_package 'Runtime Wllama' '@wllama/wllama' "$WLLAMA_VERSION" \
      "$WLLAMA_DIR" 'esm/index.js,esm/wasm/wllama.wasm' \
      WLLAMA_TARBALL WLLAMA_INTEGRITY \
      'package/esm/index.js:esm/index.js' \
      'package/esm/wasm/wllama.wasm:esm/wasm/wllama.wasm' \
      'package/LICENSE:LICENSE' \
      'package/LICENCE:LICENCE' \
      'package/README.md:README.md' || failures=$((failures + 1))

    install_npm_package 'Runtime Wllama compat' '@wllama/wllama-compat' "$WLLAMA_COMPAT_VERSION" \
      "$WLLAMA_COMPAT_DIR" 'wasm/wllama.js,wasm/wllama.wasm' \
      WLLAMA_COMPAT_TARBALL WLLAMA_COMPAT_INTEGRITY \
      'package/wasm/wllama.js:wasm/wllama.js' \
      'package/wasm/wllama.wasm:wasm/wllama.wasm' \
      'package/LICENSE:LICENSE' \
      'package/LICENCE:LICENCE' \
      'package/README.md:README.md' || failures=$((failures + 1))
  fi

  if (( SKIP_WEBLLM )); then
    log_step 'Runtime WebLLM: saltato (--skip-webllm / --skip-runtime)'
  else
    install_npm_package 'Runtime WebLLM (backend GPU)' '@mlc-ai/web-llm' "$WEBLLM_VERSION" \
      "$WEBLLM_DIR" 'lib/index.js' \
      WEBLLM_TARBALL WEBLLM_INTEGRITY \
      'package/lib/index.js:lib/index.js' \
      'package/LICENSE:LICENSE' \
      'package/package.json:package.json' \
      'package/README.md:README.md' || failures=$((failures + 1))
  fi

  if (( SKIP_MODEL )); then
    log_step 'Modello GGUF: saltato (--skip-model)'
  else
    download_model || failures=$((failures + 1))
  fi

  (( CHECK_ONLY )) && { log ''; return 0; }

  if (( failures > 0 )); then
    log ''
    log '❌ Installazione incompleta:'
    log "   • $failures sezione/i fallita/e (dettagli sopra)."
    log '  Rilancia il comando: i download già completati vengono saltati.'
    log ''
    return 1
  fi

  (( SKIP_WLLAMA ))  && log $'\n⚠ Runtime Wllama saltato: il backend CPU non sarà disponibile.'
  (( SKIP_WEBLLM ))  && log $'\n⚠ Runtime WebLLM saltato: il backend GPU userà il CDN.'
  if (( SKIP_MODEL )); then
    log $'\n⚠ Modello GGUF saltato su richiesta (--skip-model).'
    log '  Esegui il comando senza --skip-model prima di aprire index.html.'
  fi

  if (( SKIP_WLLAMA || SKIP_WEBLLM || SKIP_MODEL )); then
    log ''
    return 0
  fi

  log ''
  log '✅ Tutto pronto: avvia il server con "bun run serve" (o "node server.js").'
  log '   L’applicazione ora carica solo file locali: nessuna CDN o API esterna.'
  log '   Backend GPU: vendor/web-llm/lib/index.js (i pesi MLC li scarica il browser).'
  log ''
}

main "$@"
