FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-full \
    python3-dev \
    git \
    curl \
    libfftw3-dev

COPY --from=ghcr.io/astral-sh/uv:0.7.11 /uv /uvx /bin/
