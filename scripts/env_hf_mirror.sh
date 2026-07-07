# Source before HuggingFace / datasets downloads on servers without direct HF access.
#   source scripts/env_hf_mirror.sh
#
# Override mirror URL:
#   export HF_MIRROR_URL=https://hf-mirror.com

export HF_MIRROR_URL="${HF_MIRROR_URL:-https://hf-mirror.com}"
export HF_ENDPOINT="${HF_ENDPOINT:-${HF_MIRROR_URL}}"
export HUGGINGFACE_HUB_ENDPOINT="${HUGGINGFACE_HUB_ENDPOINT:-${HF_MIRROR_URL}}"
