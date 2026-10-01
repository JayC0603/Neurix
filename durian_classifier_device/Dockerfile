FROM python:3.13-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/mobilenetv1_best.keras \
    AUDIO_MODEL_PATH=/app/models/model-audio.pt \
    IMAGE_MODEL_PYTHON=/usr/local/bin/python \
    AUDIO_MODEL_PYTHON=/usr/local/bin/python \
    CAMERA_ROI_PATH=/app/config/camera_roi.json \
    CAPTURE_OUTPUT_DIR=/app/captures

RUN apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
        alsa-utils \
        build-essential \
        ca-certificates \
        libasound2 \
        libgomp1 \
        libportaudio2 \
        libsndfile1 \
        python3-dev \
        swig \
        v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Raspberry Pi's current archive key has a legacy SHA-1 binding that Debian
# Trixie's sqv policy rejects after 2026-02-01. Scope the trust exception to
# this single HTTPS repository; Debian's repositories remain signature-checked.
RUN echo "deb [arch=arm64 trusted=yes] https://archive.raspberrypi.com/debian/ trixie main" \
        > /etc/apt/sources.list.d/raspberrypi.list

RUN apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install -y --no-install-recommends liblgpio-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY durian_classifier_device/requirements.txt ./requirements.txt
COPY durian_classifier_device/requirements-docker.txt ./requirements-docker.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-docker.txt

COPY model/MobileNetV1/mobilenetv1_best.keras /app/models/mobilenetv1_best.keras
COPY model/model-audio.pt /app/models/model-audio.pt

COPY durian_classifier_device/app.py durian_classifier_device/config.py ./
COPY durian_classifier_device/config ./config
COPY durian_classifier_device/hardware ./hardware
COPY durian_classifier_device/inference ./inference
COPY durian_classifier_device/services ./services
COPY durian_classifier_device/utils ./utils
COPY durian_classifier_device/scripts ./scripts
COPY durian_classifier_device/tests ./tests
COPY durian_classifier_device/README.md ./README.md
COPY durian_classifier_device/.env.example ./.env.example

RUN mkdir -p /app/captures /app/logs \
    && python -m compileall -q /app

EXPOSE 8081

CMD ["python", "app.py"]
