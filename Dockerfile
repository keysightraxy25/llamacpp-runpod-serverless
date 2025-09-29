FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    HF_HOME=/models/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

EXPOSE 8080

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3-pip python3.12-dev \
      git ca-certificates wget libopenblas-dev \
      build-essential pkg-config cmake ninja-build && \
    rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv $VIRTUAL_ENV && \
    $VIRTUAL_ENV/bin/pip install --upgrade pip

RUN export CUDA_STUBS=/usr/local/cuda/targets/x86_64-linux/lib/stubs; \
    ln -sf ${CUDA_STUBS}/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1; \
    CMAKE_ARGS="-DGGML_CUDA=on -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DCMAKE_CUDA_ARCHITECTURES=all" pip install --verbose llama-cpp-python; \
    rm -rf ${CUDA_STUBS}/libcuda.so

RUN pip install --no-cache-dir runpod

#RUN wget https://huggingface.co/unsloth/gemma-3-270m-it-qat-GGUF/resolve/main/gemma-3-270m-it-qat-UD-Q8_K_XL.gguf
#RUN wget https://huggingface.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF/resolve/main/DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf
#RUN wget https://huggingface.co/unsloth/gemma-3-27b-it-qat-GGUF/resolve/main/gemma-3-27b-it-qat-UD-Q8_K_XL.gguf
#RUN wget https://huggingface.co/unsloth/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-Q8_0.gguf
#RUN wget https://huggingface.co/ggml-org/Qwen3-30B-A3B-Instruct-2507-Q8_0-GGUF/resolve/main/qwen3-30b-a3b-instruct-2507-q8_0.gguf

COPY handle.py test_input.json start.sh /
RUN chmod +x /start.sh
ENTRYPOINT /start.sh
