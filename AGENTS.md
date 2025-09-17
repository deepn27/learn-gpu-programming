Here’s a concise **AGENTS.md** file you can just drop in. It only covers **problem description, environment, and installation** so Codex/agents can bootstrap and run your tasks on GPU.

---

# AGENTS.md

## Problem Definition

We want to benchmark and optimize a **fused MLP kernel** on an **NVIDIA A100 GPU**.
The computation is:

```
Y = Linear(X, W1)        # no bias
Y = SiLU(Y)
Z = Linear(Y, W2)        # no bias
```

All matrix dimensions are **4096 × 4096**.

Tasks:

1. **Baseline (PyTorch)**: implement in pure PyTorch and collect P50, P99 latencies.
2. **CUTLASS fused kernel (CUDA C++ extension)**: implement GEMM1 → SiLU → GEMM2 using CUTLASS.
3. **Native CUDA fused kernel (CUDA C++ extension)**: implement same fused sequence manually.

---

## Environment

* **Hardware**: NVIDIA A100 (SM80 architecture).
* **OS**: Ubuntu 20.04 or 22.04.
* **Driver**: NVIDIA driver ≥ 535.
* **CUDA**: CUDA 12.x (comes with container).
* **Python**: 3.10+.
* **PyTorch**: NVIDIA NGC PyTorch container (recommended).

---

## Installation Instructions

### 1. Install NVIDIA Container Runtime

```bash
# Install Docker
curl https://get.docker.com | sh
sudo systemctl --now enable docker

# Install NVIDIA container toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2. Pull and Run NGC PyTorch Container

```bash
docker pull nvcr.io/nvidia/pytorch:24.06-py3

# Mount your repo (with AGENTS.md inside) at /workspace
docker run --gpus all -it --rm \
  --shm-size=16g \
  -v $PWD:/workspace \
  nvcr.io/nvidia/pytorch:24.06-py3 /bin/bash
```

### 3. Inside the Container — Python Dependencies

```bash
cd /workspace
pip install -U pip setuptools wheel
pip install numpy tabulate json5
```

### 4. Get CUTLASS

```bash
cd /workspace
git clone --depth=1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git fused_cutlass/cutlass
```

---

✅ At this point the environment is ready:

* PyTorch baseline runs directly.
* CUTLASS and native CUDA C++ fused kernels can be built with **CMake** (pointing at `torch`’s CMake config).

---

Do you also want me to expand this into a **minimal repo skeleton** (with CMakeLists and placeholder files) so Codex can immediately start filling in code?

