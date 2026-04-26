#!/usr/bin/env bash
set -euo pipefail

# Build scry and set up its Python environment.
#
# Usage:
#   ./setup.sh           # build + venv + model dirs
#   ./setup.sh --install # also register with omegon (omegon extension install .)
#
# After setup, register with omegon:
#   omegon extension install /path/to/scry

SCRY_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${HOME}/.scry/venv"
MODELS_DIR="${HOME}/.scry/models"

# ── Ensure cargo is on PATH ───────────────────────────────────────
if ! command -v cargo &>/dev/null && [ -x "${HOME}/.cargo/bin/cargo" ]; then
    export PATH="${HOME}/.cargo/bin:${PATH}"
fi

if ! command -v cargo &>/dev/null; then
    echo "ERROR: cargo not found. Install Rust: https://rustup.rs"
    echo "  macOS (nex):       nex install rustup"
    echo "  macOS (Homebrew):  brew install rustup && rustup-init"
    echo "  Linux:             curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    echo ""
    echo "  After installing, make sure ~/.cargo/bin is in your PATH."
    exit 1
fi

# ── Find compatible Python (>= 3.10) ──────────────────────────────
python_version_ok() {
    "$1" -c 'import sys; exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null
}

find_python() {
    # Prefer explicitly-versioned binaries (highest first), then bare python3
    for candidate in python3.12 python3.11 python3.10 python3.13 python3; do
        if command -v "$candidate" &>/dev/null && python_version_ok "$candidate"; then
            echo "$candidate"
            return
        fi
    done
    echo ""
}

PYTHON_BIN="${SCRY_PYTHON:-$(find_python)}"
if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: No compatible Python found (need >= 3.10)."
    echo "  Available: $(python3 --version 2>&1 || echo 'none')"
    echo "  Install Python 3.12:"
    echo "    nex:       nex install python312"
    echo "    Homebrew:  brew install python@3.12"
    exit 1
fi
echo "==> Using Python: $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"

# ── Build ───────────────────────────────────────────────────────────
echo "==> Building scry (release)..."
cargo build --release --manifest-path "${SCRY_DIR}/Cargo.toml"

# ── Python venv ─────────────────────────────────────────────────────
VENV_PYTHON="${VENV_DIR}/bin/python"
recreate_venv=false
if [ ! -x "$VENV_PYTHON" ]; then
    recreate_venv=true
elif ! "$VENV_PYTHON" -c "import sys" &>/dev/null; then
    # venv python is broken (e.g. system python was upgraded/removed)
    recreate_venv=true
else
    # Recreate if the base python major.minor changed
    venv_ver="$("$VENV_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "")"
    base_ver="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "")"
    if [ "$venv_ver" != "$base_ver" ]; then
        echo "==> Python version changed ($venv_ver -> $base_ver), recreating venv..."
        recreate_venv=true
    fi
fi

if [ "$recreate_venv" = true ]; then
    echo "==> Creating Python venv at ${VENV_DIR}..."
    rm -rf "${VENV_DIR}"
    "$PYTHON_BIN" -m venv "${VENV_DIR}"
fi

echo "==> Installing Python dependencies..."
"${VENV_DIR}/bin/pip" install --quiet --upgrade pip
"${VENV_DIR}/bin/pip" install --quiet -r "${SCRY_DIR}/requirements.txt"

if ! "${VENV_DIR}/bin/python" -c "import torch; print(f'  torch {torch.__version__} mps={torch.backends.mps.is_available()}')" 2>&1; then
    echo "WARNING: torch import failed. The worker will not start."
    echo "  Try: ${VENV_DIR}/bin/pip install torch --force-reinstall"
fi

# ── Model directories ──────────────────────────────────────────────
echo "==> Creating model directories at ${MODELS_DIR}..."
mkdir -p "${MODELS_DIR}/checkpoints"
mkdir -p "${MODELS_DIR}/loras"
mkdir -p "${MODELS_DIR}/vae"
mkdir -p "${MODELS_DIR}/upscale_models"
mkdir -p "${MODELS_DIR}/clip"
mkdir -p "${MODELS_DIR}/controlnet"
mkdir -p "${MODELS_DIR}/unet"
mkdir -p "${HOME}/.scry/output"

# ── Optional: register with omegon ─────────────────────────────────
if [ "${1:-}" = "--install" ]; then
    echo "==> Registering with omegon..."
    if command -v omegon &>/dev/null; then
        omegon extension install "${SCRY_DIR}" || true
    else
        echo "  omegon not found in PATH. Register manually:"
        echo "  omegon extension install ${SCRY_DIR}"
    fi
fi

echo ""
echo "Ready."
echo "  Binary:  ${SCRY_DIR}/target/release/scry"
echo "  Venv:    ${VENV_DIR} ($($VENV_DIR/bin/python --version 2>&1))"
echo "  Models:  ${MODELS_DIR}/"
echo "  Output:  ~/.scry/output/"
echo ""
echo "Model discovery (automatic):"
echo "  ~/.scry/models/          your models"
echo "  ~/.cache/huggingface/    HF Hub cache"
echo "  ~/ComfyUI/models/        if present"
echo "  ~/stable-diffusion-webui/ if present"
echo ""
echo "Register with omegon:"
echo "  omegon extension install ${SCRY_DIR}"
