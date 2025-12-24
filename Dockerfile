FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

WORKDIR /workspace

# Broad GPU arch coverage (Maxwell -> Blackwell)
ENV TORCH_CUDA_ARCH_LIST="5.2;6.0;6.1;7.0;7.5;8.0;8.6;8.7;8.9;9.0"
# Optional: speed up builds
ENV MAX_JOBS=8
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1

# System deps in one layer (+ cleanup)
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      libgl1 \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python deps in fewer layers
RUN pip install --upgrade pip && \
    pip install \
      pyhocon open3d scipy pandas trimesh \
      PyMCubes \
      libigl \
      opencv-python \
      python-igraph \
      "git+https://github.com/cong-yi/DualMesh-UDF" \
      spconv-cu126

# Copy extensions late for better Docker cache reuse
COPY extensions ./extensions

# Force rebuild from scratch: remove old build artifacts and build dir
RUN rm -rf \
      ./extensions/chamfer_dist/build \
      ./extensions/chamfer_dist/*.so \
      ./extensions/chamfer_dist/**/*.so

RUN pip  install --no-build-isolation ./extensions/chamfer_dist

RUN pip  install comet_ml \
    py-cpuinfo