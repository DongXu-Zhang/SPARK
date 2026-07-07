#!/usr/bin/env bash
# Shared helpers: live nohup logs + one-glance status file.
# shellcheck shell=bash

progress_now() { date -Iseconds; }

# Write human-readable status (overwrite each call).
# Usage: progress_set_status MODEL PHASE STEP DETAIL_LOG
progress_set_status() {
  local model="${1:-?}" phase="${2:-?}" step="${3:-?}" detail="${4:-}"
  local f="${PROGRESS_STATUS_FILE:-logs/pipeline_status.txt}"
  mkdir -p "$(dirname "$f")"
  cat >"$f" <<EOF
updated: $(progress_now)
model: ${model}
phase: ${phase}
step: ${step}
detail_log: ${detail}
live_log: ${PROGRESS_LIVE_LOG:-}
EOF
}

# Prefix + tee to detail log and live log (line-buffered).
# Usage: progress_run_tee "PREFIX" detail.log live.log -- command args...
progress_run_tee() {
  local prefix="$1" detail_log="$2" live_log="$3"
  shift 3
  [[ "${1:-}" == "--" ]] && shift
  mkdir -p "$(dirname "$detail_log")" "$(dirname "$live_log")"

  echo "[$(progress_now)] ${prefix} >>> START" | tee -a "${detail_log}" -a "${live_log}"
  set +e
  stdbuf -oL -eL "$@" 2>&1 | while IFS= read -r line; do
    printf '[%s] %s\n' "${prefix}" "${line}" | tee -a "${detail_log}" -a "${live_log}" >/dev/null
  done
  local rc=${PIPESTATUS[0]}
  set -e
  echo "[$(progress_now)] ${prefix} <<< END exit=${rc}" | tee -a "${detail_log}" -a "${live_log}"
  return "${rc}"
}

progress_banner() {
  local msg="$1" live_log="${2:-${PROGRESS_LIVE_LOG:-/dev/null}}"
  mkdir -p "$(dirname "$live_log")"
  {
    echo ""
    echo "================================================================"
    echo "[$(progress_now)] ${msg}"
    echo "================================================================"
    echo ""
  } | tee -a "${live_log}"
}
