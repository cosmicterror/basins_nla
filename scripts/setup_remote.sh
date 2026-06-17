#!/usr/bin/env bash
# Build the SGLang + Gemma-3 NLA serving environment ON a vast.ai box.
# Run from the uploaded project dir after /gpu setup has put files + HF token in place.
#
#   bash scripts/setup_remote.sh
#
# Idempotent-ish: the patcher skips already-patched files; pip installs are no-ops
# if satisfied. Expects the pytorch base image (torch already present).
set -euo pipefail

KITFT="${KITFT:-/workspace/natural_language_autoencoders}"
# Pin to the SGLang version the NLA patches were verified against (docs/inference.md).
# A newer SGLang may move the anchors the patcher keys on -> its asserts will fail loudly.
SGLANG_VER="${SGLANG_VER:-0.5.6}"

echo "### 1/5  base deps (torch/CUDA come from the image)"
# libnuma1 provides libnuma.so.1, which sgl_kernel's compiled SM90 ops link
# against but the pytorch base image does not ship. Without it sglang dies at
# import: 'ImportError: libnuma.so.1: cannot open shared object file'.
apt-get update -qq && apt-get install -y libnuma1 ffmpeg   # ffmpeg: libtorchcodec for sentence-transformers
pip uninstall -y torchvision || true          # /gpu note: avoids transformer-lens conflict
pip install "transformers>=4.50" accelerate safetensors httpx orjson pyyaml numpy pyarrow \
            anthropic   # anthropic for measure.py judge (embedder uses transformers directly)

echo "### 2/5  sglang==${SGLANG_VER} (verified version for the NLA patches)"
pip install "sglang[all]==${SGLANG_VER}"

echo "### 3/5  kitft repo (for patches/ + apply script)"
[ -d "$KITFT" ] || git clone --depth 1 \
    https://github.com/kitft/natural_language_autoencoders.git "$KITFT"

echo "### 4/5  apply NLA SGLang patches to the INSTALLED package"
# apply_sglang_patches.sh expects a source tree at \$SRC/python/sglang/srt, but a
# pip install lives at site-packages/sglang/srt (no python/ prefix). Bridge with a
# symlink shim so the patcher edits the installed files in place.
SGPKG="$(python -c 'import sglang, os; print(os.path.dirname(sglang.__file__))')"
SHIM=/workspace/_sglang_shim
mkdir -p "$SHIM/python"
ln -sfn "$SGPKG" "$SHIM/python/sglang"
bash "$KITFT/patches/apply_sglang_patches.sh" "$SHIM"

echo "### 5/5  HF gated-access check (Gemma is gated)"
python - <<'PY'
import pathlib
t = pathlib.Path.home() / ".cache/huggingface/token"
print("HF token present ✓" if t.exists()
      else "WARNING: no HF token at ~/.cache/huggingface/token — gated Gemma will 401")
PY

echo "setup_remote.sh: done. Next -> bash scripts/serve_av.sh <fa3|triton>"
