ARG CUDA
FROM nvidia/cuda:${CUDA}-cudnn-devel-ubuntu24.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-full \
    python3-dev \
    git \
    curl \
    libfftw3-dev

COPY --from=ghcr.io/astral-sh/uv:0.10.8 /uv /uvx /bin/
