#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "Created .env from .env.example"
fi

# .env only contains shell-compatible KEY=value entries. Relative paths are
# resolved from the application directory, regardless of where the repo lives.
set -a
source .env
set +a
MODEL_PATH="${MODEL_PATH:-../model/onnx/mobilenetv1_image.onnx}"
AUDIO_MODEL_PATH="${AUDIO_MODEL_PATH:-../model/model-audio.pt}"
[[ "$MODEL_PATH" = /* ]] || MODEL_PATH="$PROJECT_DIR/$MODEL_PATH"
[[ "$AUDIO_MODEL_PATH" = /* ]] || AUDIO_MODEL_PATH="$PROJECT_DIR/$AUDIO_MODEL_PATH"

echo "CPU architecture: $(uname -m)"
echo "Python: $(python3 --version 2>&1)"

if grep -q '^lgpio' requirements.txt && ! command -v swig >/dev/null 2>&1; then
    echo "ERROR: swig is required to build the lgpio Python package on this system." >&2
    echo "Install it first, then rerun this script:" >&2
    echo "  sudo apt update" >&2
    echo "  sudo apt install -y swig" >&2
    exit 3
fi

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "ERROR: image model not found: $MODEL_PATH" >&2
    echo "Set MODEL_PATH in $PROJECT_DIR/.env" >&2
    exit 1
fi
if [[ ! -f "$AUDIO_MODEL_PATH" ]]; then
    echo "ERROR: audio model not found: $AUDIO_MODEL_PATH" >&2
    echo "Set AUDIO_MODEL_PATH in $PROJECT_DIR/.env" >&2
    exit 1
fi

mkdir -p captures logs
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

case "${MODEL_PATH##*.}" in
    onnx)
        python inference/onnx_image_model_worker.py --model "$MODEL_PATH" </dev/null
        ;;
    keras)
        python - "$MODEL_PATH" <<'PY'
import sys
import tensorflow as tf

model = tf.keras.models.load_model(sys.argv[1])
print("TensorFlow:", tf.__version__)
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)
if int(model.output_shape[-1]) != 3:
    raise SystemExit("ERROR: expected exactly 3 model outputs")
PY
        ;;
    pt|pth)
        python inference/torch_image_model_worker.py --model "$MODEL_PATH" </dev/null
        ;;
    *)
        echo "ERROR: unsupported image model extension: $MODEL_PATH" >&2
        exit 2
        ;;
esac

python -c 'import torch, librosa; print("PyTorch:", torch.__version__, "librosa:", librosa.__version__)'

if [[ -e /dev/i2c-1 ]]; then
    echo "I2C bus /dev/i2c-1 is available"
else
    echo "WARNING: /dev/i2c-1 is missing. Enable I2C manually with sudo raspi-config."
    echo "Interface Options -> I2C -> Enable"
fi

if command -v i2cdetect >/dev/null 2>&1; then
    echo "i2cdetect is installed"
else
    echo "WARNING: i2cdetect is missing; install i2c-tools manually."
fi

echo "Setup checks completed."
