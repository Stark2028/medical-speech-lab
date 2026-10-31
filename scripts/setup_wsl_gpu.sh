#!/usr/bin/env bash
# Run inside Ubuntu WSL. Installs a project-specific Python 3.12 GPU environment.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
uv_bin="$HOME/.local/bin/uv"
python_bin="$HOME/.venvs/medspeech/bin/python"

if [[ ! -x "$uv_bin" ]]; then
  installer="$(mktemp)"
  trap 'rm -f "$installer"' EXIT
  curl -fLsS https://astral.sh/uv/0.13.6/install.sh -o "$installer"
  UV_UNMANAGED_INSTALL="$HOME/.local/bin" sh "$installer"
fi

"$uv_bin" python install 3.12
"$uv_bin" venv "$HOME/.venvs/medspeech" --python 3.12
"$uv_bin" pip install --python "$python_bin" torch==2.12.0
"$uv_bin" pip install --python "$python_bin" -r "$repo_root/requirements.txt"
"$python_bin" -c 'import torch; print("PyTorch", torch.__version__, "CUDA", torch.cuda.is_available()); assert torch.cuda.is_available(), "CUDA is unavailable"'
