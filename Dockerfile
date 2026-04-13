FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

USER root
ARG DEBIAN_FRONTEND=noninteractive

LABEL github_repo="https://github.com/SWivid/F5-TTS"

RUN set -x \
    && apt-get update \
    && apt-get -y install \
        wget curl man git less openssl libssl-dev unzip unar \
        build-essential aria2 tmux vim openssh-server \
        sox libsox-fmt-all libsox-fmt-mp3 libsndfile1-dev \
        ffmpeg \
        libavcodec-dev libavformat-dev libavdevice-dev \
        libavutil-dev libavfilter-dev libswscale-dev libswresample-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /workspace/F5-TTS

COPY . .

RUN pip install --no-cache-dir -e .[eval] \
    && pip install --no-cache-dir -e ./kugelaudio

ENV SHELL=/bin/bash

