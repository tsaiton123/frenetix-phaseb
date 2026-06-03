FROM --platform=linux/amd64 ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        python3-pip \
        build-essential \
        cmake \
        git \
        libeigen3-dev \
        libboost-all-dev \
        libomp-dev \
        libgl1 \
        libglib2.0-0 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml poetry.lock README.md ./
COPY frenetix_rl ./frenetix_rl

RUN pip install .

RUN pip install "matplotlib<3.8" "scipy<1.14"

COPY configurations ./configurations
COPY scenarios ./scenarios
COPY scenarios_validation ./scenarios_validation
COPY scenarios_test ./scenarios_test
COPY analysis ./analysis
COPY scripts ./scripts
COPY train.py execute.py \
     run_phase_b_eval.py run_hp_100k_eval.py run_700k_eval.py \
     run_default_planner.py run_hybrid_planner.py ./

# logs/ is created at runtime by train.py; no model ships in this repo.
CMD ["python", "train.py"]
