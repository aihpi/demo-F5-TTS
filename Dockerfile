# CUDA 13.1 + cuDNN 9 — matches node driver 590.48.01 on gx16 (A30, compute 8.0)
FROM pytorch/pytorch:2.7.0-cuda13.1-cudnn9-devel

USER root

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_VERSION=2.7.0
ARG CUDA_TAG=cu131

LABEL github_repo="https://github.com/SWivid/F5-TTS"

RUN set -x \
    && apt-get update \
    && apt-get -y install wget curl man git less openssl libssl-dev unzip unar build-essential aria2 tmux vim \
    && apt-get install -y openssh-server sox libsox-fmt-all libsox-fmt-mp3 libsndfile1-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /workspace

RUN git clone https://github.com/aihpi/F5-TTS.git \
    && cd F5-TTS \
    && pip install -e .[eval]

# Force-reinstall torch + torchaudio from the official CUDA 13.1 wheel index so they
# share the same ABI regardless of what pip resolved above.
RUN pip install --no-cache-dir --force-reinstall \
    torch==${TORCH_VERSION} \
    torchaudio==${TORCH_VERSION} \
    --index-url https://download.pytorch.org/whl/${CUDA_TAG}

ENV SHELL=/bin/bash

WORKDIR /workspace/F5-TTS
