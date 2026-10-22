# Development container for the R torch package, with PyTorch for comparing results.
# The libtorch version must match `torch_version` in R/install.R.
FROM docker.io/rocker/r-ver:4.5.1

ARG TORCH_VERSION=2.8.0

ENV DEBIAN_FRONTEND=noninteractive \
    MAKEFLAGS=-j16 \
    TORCH_HOME=/opt/torch-deps \
    TORCH_INSTALL=1 \
    CUDA=cpu \
    TORCH_COMMIT_SHA=none \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl unzip cmake libcurl4-openssl-dev libssl-dev libxml2-dev libgit2-dev \
      python3 python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# PyTorch (CPU), to compare R torch against
RUN python3 -m venv $VIRTUAL_ENV \
    && pip install --no-cache-dir torch==${TORCH_VERSION} numpy --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir safetensors

# R dependencies of torch (binaries from P3M, as configured by rocker), plus dev tools
COPY DESCRIPTION /tmp/torch-desc/DESCRIPTION
RUN Rscript -e 'install.packages(c("remotes", "devtools", "testthat", "reticulate"))' \
    && Rscript -e 'remotes::install_deps("/tmp/torch-desc", dependencies = TRUE)' \
    && rm -rf /tmp/torch-desc

# libtorch and lantern (of the development version on main), installed once into TORCH_HOME. The package itself is installed from the
# mounted source (see post-create.sh), which then finds the dependencies here.
RUN mkdir -p $TORCH_HOME \
    && Rscript -e 'install.packages("torch")' \
    && LANTERN_URL=https://torch-cdn.mlverse.org/binaries/refs/heads/main/latest/lantern-0.17.0.9000%2Bcpu%2Bx86_64-Linux.zip \
       TORCH_URL=https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${TORCH_VERSION}%2Bcpu.zip \
       Rscript -e 'torch::install_torch(.inform_restart = FALSE)' \
    && ls $TORCH_HOME/lib

WORKDIR /workspaces/torch
