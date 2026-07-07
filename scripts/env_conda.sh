#!/usr/bin/env bash
# Activate project conda env for nohup / batch jobs.
# Override: CONDA_ENV=other_env

CONDA_ENV="${CONDA_ENV:-frontier}"

if [[ -n "${CONDA_DEFAULT_ENV:-}" && "${CONDA_DEFAULT_ENV}" == "${CONDA_ENV}" ]]; then
  return 0 2>/dev/null || true
fi

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  echo "[env_conda] cannot find conda.sh" >&2
  exit 1
fi

conda activate "${CONDA_ENV}"
