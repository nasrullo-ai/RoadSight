# Clean-machine image for the scored run (SPEC section 14).
#   docker build -t roadsight .
#   docker run --gpus all --network none -v $PWD/data/samples:/videos -v $PWD/outputs:/out roadsight
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.10 python3-pip libgl1 libglib2.0-0 \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt

COPY . .
# Import once at build time so the image is known-good before any network is removed.
RUN python -c "import solution; print('solution import ok')"

CMD ["python", "run_submission.py", "--videos", "/videos", "--out", "/out/predictions.json"]
