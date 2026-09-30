#!/usr/bin/env bash
# Configure the CUDA toolkit tree that numba needs to JIT-compile kernels.
#
# The nvidia-* pip wheels ship an UNVERSIONED libnvvm.so, but numba's finder
# (numba.misc.findlib) only matches `lib<name>.so.<digits>`. Without the
# versioned symlink every kernel launch fails with:
#   NvvmSupportError: libNVVM cannot be found.
#
# This assembles a CUDA_HOME-shaped tree from the installed wheels and prints
# the exports. Idempotent: safe to re-run.
set -euo pipefail

VENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv"
CUDA_HOME_DIR="$VENV_DIR/cuda_home"
PY="$VENV_DIR/bin/python"

[ -x "$PY" ] || { echo "error: no interpreter at $PY" >&2; exit 1; }

SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
NVCC="$SITE/nvidia/cuda_nvcc"
RUNTIME="$SITE/nvidia/cuda_runtime"
NVRTC="$SITE/nvidia/cuda_nvrtc"

[ -f "$NVCC/nvvm/lib64/libnvvm.so" ] || {
  echo "error: libnvvm.so not found under $NVCC" >&2
  echo "       run: $PY -m pip install -r requirements.txt" >&2
  exit 1
}

mkdir -p "$CUDA_HOME_DIR/nvvm" "$CUDA_HOME_DIR/lib64"
cp -rn "$NVCC/nvvm/lib64" "$NVCC/nvvm/libdevice" "$CUDA_HOME_DIR/nvvm/" 2>/dev/null || true

# The load-bearing step: give libnvvm.so the versioned soname numba looks for.
ln -sf libnvvm.so "$CUDA_HOME_DIR/nvvm/lib64/libnvvm.so.12"

# cudart + nvrtc, with unversioned names for ctypes
for so in "$RUNTIME"/lib/libcudart.so.12*; do
  [ -e "$so" ] && cp -n "$so" "$CUDA_HOME_DIR/lib64/" 2>/dev/null || true
done
ln -sf libcudart.so.12 "$CUDA_HOME_DIR/lib64/libcudart.so"
for so in "$NVRTC"/lib/libnvrtc.so.*; do
  [ -e "$so" ] && cp -n "$so" "$CUDA_HOME_DIR/lib64/" 2>/dev/null || true
done

echo "export CUDA_HOME=\"$CUDA_HOME_DIR\""
echo "export LD_LIBRARY_PATH=\"$CUDA_HOME_DIR/lib64\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\""
