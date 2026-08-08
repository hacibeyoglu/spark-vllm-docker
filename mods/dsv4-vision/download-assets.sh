#!/usr/bin/env bash
# Host-side: download the DSV4 vision encoder (tower + adapter) into the HF
# cache tree so it is visible to every node / container via the standard
# spark-vllm-docker HF mount (/root/.cache/huggingface).
#
# This is a separate step from the recipe's `model:` download because the
# backbone and the vision encoder live in two different HF repos.
#
# Usage:
#   ./mods/dsv4-vision/download-assets.sh
#
# Env:
#   HF_CACHE      host HF cache dir (default ~/.cache/huggingface)
#   HF_TOKEN      optional auth token for HF Hub
#   VISION_REPO   HF repo with tower+adapter (default FlyCockpit/...-vision)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
VISION_REPO="${VISION_REPO:-FlyCockpit/DeepSeek-V4-Flash-0731-vision}"
ASSETS_DIR="$HF_CACHE/dsv4-vision-assets"
TOWER_DIR="$ASSETS_DIR/tower"
CKPT_DIR="$ASSETS_DIR/adapter"

ADAPTER_MD5_EXPECT="${ADAPTER_MD5_EXPECT:-d9b3b3bda8f790ecf7cd5a98e6fb93a5}"
TOWER_MD5_EXPECT="${TOWER_MD5_EXPECT:-2d5dba626d816cc367d28b32e744830e}"

export HF_HOME="${HF_HOME:-$HF_CACHE}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

if [[ -z "${HF_TOKEN:-}" && -n "${HF_TOKEN_FILE:-}" && -f "$HF_TOKEN_FILE" ]]; then
  HF_TOKEN="$(tr -d '\r\n' < "$HF_TOKEN_FILE")"
  export HF_TOKEN
fi

mkdir -p "$TOWER_DIR" "$CKPT_DIR"

if command -v hf >/dev/null 2>&1; then
  HF=(hf)
elif [[ -x "$HOME/.local/bin/hf" ]]; then
  HF=("$HOME/.local/bin/hf")
else
  HF=(python3 -m huggingface_hub.cli.hf)
fi

echo "== download vision encoder: $VISION_REPO =="
"${HF[@]}" download "$VISION_REPO" \
  tower/deepencoder_v2_tower.safetensors \
  --local-dir "$ASSETS_DIR"
"${HF[@]}" download "$VISION_REPO" \
  adapter/merged-004800-5af0c5.pt \
  adapter/latest.pt \
  --local-dir "$ASSETS_DIR"

ADAPTER_FILE="$ASSETS_DIR/adapter/merged-004800-5af0c5.pt"
TOWER_FILE="$ASSETS_DIR/tower/deepencoder_v2_tower.safetensors"
[[ -f "$ADAPTER_FILE" ]] || { echo "missing $ADAPTER_FILE" >&2; exit 1; }
[[ -f "$TOWER_FILE" ]] || { echo "missing $TOWER_FILE" >&2; exit 1; }

ln -sfn "$(basename "$ADAPTER_FILE")" "$(dirname "$ADAPTER_FILE")/latest.pt"

echo "== md5 verify =="
adapter_md5=$(md5sum "$ADAPTER_FILE" | awk '{print $1}')
tower_md5=$(md5sum "$TOWER_FILE" | awk '{print $1}')
echo "adapter $ADAPTER_FILE"
echo "  md5 $adapter_md5 (expect $ADAPTER_MD5_EXPECT)"
echo "tower   $TOWER_FILE"
echo "  md5 $tower_md5 (expect $TOWER_MD5_EXPECT)"
[[ "$adapter_md5" == "$ADAPTER_MD5_EXPECT" ]] || { echo "ADAPTER MD5 MISMATCH" >&2; exit 1; }
[[ "$tower_md5" == "$TOWER_MD5_EXPECT" ]] || { echo "TOWER MD5 MISMATCH" >&2; exit 1; }

echo "DOWNLOAD_OK"
echo "  adapter -> $ADAPTER_FILE"
echo "  tower   -> $TOWER_FILE"
