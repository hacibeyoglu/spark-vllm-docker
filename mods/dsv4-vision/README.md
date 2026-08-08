# dsv4-vision mod

Adds **vision** to `deepseek-ai/DeepSeek-V4-Flash-0731` on the b12x serving
stack, by migrating the vLLM plugin from
[`DeepSeek-V4-Vision-2x-DGX-Sparks`](../../DeepSeek-V4-Vision-2x-DGX-Sparks).

It is a pure **vLLM plugin** (registered via the `vllm.general_plugins` entry
point) — no core patches. It installs the model wrapper that splices the
DeepEncoderV2 vision tower + trained adapter into the DSV4 HashRouter backbone
using vLLM's public multimodal `requires_raw_input_tokens` path.

## What this mod does (runs inside the container, per node)

1. `pip install -e` the plugin package in `plugin/` and verifies the
   `vllm.general_plugins` entry point is registered.
2. Builds the `dsv4-0731-vision` model directory (an architecture
   symlink tree over the 0731 snapshot with `architectures ->
   DeepseekV4VisionForCausalLM`) inside the mounted HF cache.
3. Verifies the vision encoder assets are present.

## Encoder assets (separate HF repo)

The **backbone** (`deepseek-ai/DeepSeek-V4-Flash-0731`) is downloaded by the
recipe's `--setup` / `hf-download.sh`. The **vision encoder** lives in a second
repo and must be downloaded on the **host** once:

```bash
./mods/dsv4-vision/download-assets.sh   # tower + adapter, md5-verified
```

This places them at
`$HOME/.cache/huggingface/dsv4-vision-assets/` so every node sees them via the
standard HF mount. If you skip it, the mod still installs and warns.

## Recipe env that must be set

```yaml
env:
  DSV4_VISION_TOWER: /root/.cache/huggingface/dsv4-vision-assets/tower/deepencoder_v2_tower.safetensors
  DSV4_VISION_ADAPTER: /root/.cache/huggingface/dsv4-vision-assets/adapter/merged-004800-5af0c5.pt
```

## Path notes

- spark-vllm-docker mounts the HF cache at `/root/.cache/huggingface`;
  `make_vision_model_dir.py` defaults its container-paths to that, and honours
  `DSV4_VISION_CONTAINER_CACHE` / `HF_CACHE` / `HF_HOME` overrides (so it also
  works unchanged on the original `/cache/huggingface` stack).
- The served model is `.../dsv4-0731-vision` (not the raw snapshot): point
  `vllm serve` at
  `/root/.cache/huggingface/dsv4-0731-vision` (or the HF-hub equivalent).
