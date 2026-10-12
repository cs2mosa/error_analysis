# =============================================================================
# Qwen3-VL-4B Error Analysis Engine
# CUDA 13.0 • Python 3.13 • PyTorch 2.14
# =============================================================================
FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04

# --------------------------------------------------------------------------- #
# 0. Hardware compatibility fix for GeForce RTX GPUs (sm_120 / RTX 5090 / RTX 3000)
# --------------------------------------------------------------------------- #
# cuda-compat is designed strictly for Datacenter GPUs (A100/H100).
# On GeForce cards, it attempts forward compatibility and crashes with CUDA Error 804.
# Removing compat libraries forces PyTorch to use the host's native NVIDIA driver.
ENV CUDNN_FORWARD_COMPAT_DISABLE=1
RUN rm -rf /usr/local/cuda/compat /usr/local/cuda-*/compat /etc/ld.so.conf.d/*cuda-compat*.conf 2>/dev/null && ldconfig || true

# --------------------------------------------------------------------------- #
# 1. OS-level packages + Python 3.13
# --------------------------------------------------------------------------- #
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.13 python3.13-dev python3.13-venv \
    curl git wget build-essential cmake pkg-config \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libgl1 libglx-mesa0 libegl1 libopengl0 \
    ffmpeg ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Create a Python 3.13 virtual environment (PEP 668 compliant)
RUN python3.13 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# --------------------------------------------------------------------------- #
# 2. Python dependencies (layer-cached before workspace copy)
# --------------------------------------------------------------------------- #
WORKDIR /workspace

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch with CUDA 13.0
ENV TORCH_CUDA_ARCH_LIST="12.0+PTX"
RUN pip install --no-cache-dir \
    torch==2.14.1+cu130 \
    torchvision==0.29.1+cu130 \
    torchaudio==2.11.0+cu130 \
    --index-url https://download.pytorch.org/whl/cu130

# Install application dependencies via requirements.txt
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# --------------------------------------------------------------------------- #
# 3. Application code
# --------------------------------------------------------------------------- #
COPY . /workspace/

# --------------------------------------------------------------------------- #
# 4. Environment variables
# --------------------------------------------------------------------------- #
ENV HF_HOME=/workspace/.hf_cache
ENV TRANSFORMERS_CACHE=/workspace/.hf_cache
ENV HF_DATASETS_CACHE=/workspace/.hf_cache/datasets

# Model repository and optional explicit path override
ENV MODEL_NAME=Qwen/Qwen3-VL-4B-Instruct
ENV MODEL_PATH=""

# Headless matplotlib rendering (no X11 / Xvfb required)
ENV MPLBACKEND=Agg

# CUDA devices
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Python package resolution for `python -m ea ...`
ENV PYTHONPATH=/workspace

# --------------------------------------------------------------------------- #
# 5. Entrypoint
# --------------------------------------------------------------------------- #
COPY docker/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--help"]
