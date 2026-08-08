#!/bin/bash
# DSV4 vision mod — runs INSIDE each container (per-node) after startup, before
# the vLLM launch script. Two responsibilities:
#
#   1. Install the vision vLLM plugin (model wrapper + DeepEncoderV2 tower) so
#      `vllm serve .../dsv4-0731-vision` builds DeepseekV4VisionForCausalLM.
#   2. Build the `dsv4-0731-vision` model directory (arch symlink tree) inside
#      the HF cache and verify the vision encoder assets are present.
#
# The vision ENCODER assets (tower + adapter) are downloaded ON THE HOST by
# `./mods/dsv4-vision/download-assets.sh` (a separate HF repo from the
# backbone). The recipe env must point at them via:
#     DSV4_VISION_TOWER=  <container cache>/dsv4-vision-assets/tower/deepencoder_v2_tower.safetensors
#     DSV4_VISION_ADAPTER=<container cache>/dsv4-vision-assets/adapter/merged-004800-5af0c5.pt
#
# Mounted cache path follows spark-vllm-docker convention:
#     $HF_CACHE_DIR:/root/.cache/huggingface   (set HF_CACHE to override)
set -euo pipefail

PREFIX="[dsv4-vision]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_CACHE="${DSV4_VISION_CONTAINER_CACHE:-/root/.cache/huggingface}"
PLUGIN_DIR="$MOD_DIR/plugin"

echo "=== DSV4 vision mod ==="

# --------------------------------------------------------------------------
# 1. Sanity: is the backbone snapshot present (needed to build the model dir)?
# --------------------------------------------------------------------------
SNAPS="$CONTAINER_CACHE/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots"
if [[ ! -d "$SNAPS" ]] || ! compgen -G "$SNAPS/*" >/dev/null; then
    echo "$PREFIX WARNING: backbone snapshot not found under '$SNAPS'." >&2
    echo "$PREFIX   Run the recipe with --setup, or ./hf-download.sh deepseek-ai/DeepSeek-V4-Flash-0731 -c" >&2
fi

# --------------------------------------------------------------------------
# 2. Install the vision plugin (editable, no deps so it never clashes).
# --------------------------------------------------------------------------
echo "$PREFIX installing vision plugin from $PLUGIN_DIR ..."
python3 -m pip install -e "$PLUGIN_DIR" --no-deps -q
python3 - <<'PY'
from importlib.metadata import entry_points
found = False
for ep in entry_points().select(group="vllm.general_plugins"):
    if ep.name == "dsv4_vision":
        found = True
print(f"[dsv4-vision] vllm.general_plugins entry point present: {found}")
if not found:
    raise SystemExit("dsv4_vision plugin entry point NOT registered")
PY

# --------------------------------------------------------------------------
# 3. Verify vision encoder assets are in the mounted cache.
# --------------------------------------------------------------------------
TOWER="$CONTAINER_CACHE/dsv4-vision-assets/tower/deepencoder_v2_tower.safetensors"
ADAPTER="$CONTAINER_CACHE/dsv4-vision-assets/adapter/merged-004800-5af0c5.pt"
if [[ ! -f "$TOWER" ]]; then
    echo "$PREFIX WARNING: vision tower not found at '$TOWER'." >&2
    echo "$PREFIX   Download on the host:  ./mods/dsv4-vision/download-assets.sh" >&2
fi
if [[ ! -f "$ADAPTER" ]]; then
    echo "$PREFIX WARNING: vision adapter not found at '$ADAPTER'." >&2
    echo "$PREFIX   Download on the host:  ./mods/dsv4-vision/download-assets.sh" >&2
fi

# --------------------------------------------------------------------------
# 4. Build the dsv4-0731-vision model dir (arch symlink tree).
# --------------------------------------------------------------------------
export DSV4_VISION_CONTAINER_CACHE="$CONTAINER_CACHE"
if [[ -n "${HF_CACHE:-}" ]]; then
    export HF_CACHE
fi
echo "$PREFIX building dsv4-0731-vision model directory ..."
python3 "$MOD_DIR/make_vision_model_dir.py"

echo "=== OK: vision plugin installed; model dir ready ==="
