FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update -y && apt-get install -y \
    sudo software-properties-common dialog apt-utils \
    tzdata locales curl wget git \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    libfontconfig1-dev libfreetype6-dev libpng-dev \
    libharfbuzz-dev libfribidi-dev libtiff5-dev libjpeg-dev \
    cmake make gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Install R via rig
RUN curl -Ls https://github.com/r-lib/rig/releases/download/latest/rig-linux-latest.tar.gz | \
    tar xz -C /usr/local && \
    rig add release && \
    ln -sf /opt/R/$(ls /opt/R)/bin/R /usr/local/bin/R && \
    ln -sf /opt/R/$(ls /opt/R)/bin/Rscript /usr/local/bin/Rscript

# Parallel compilation
RUN echo "MAKEFLAGS=-j$(nproc)" >> "$(R RHOME)/etc/Renviron.site"

# Copy source and optional lantern artifacts
COPY . /build

ARG LANTERN_BASE_URL=""
ENV LANTERN_BASE_URL=${LANTERN_BASE_URL}

ARG TORCH_URL=""
ENV TORCH_URL=${TORCH_URL}
ENV TORCH_INSTALL=1

# Install pak, R dependencies, and cudatoolkit
RUN Rscript -e "\
    install.packages('pak', repos = 'https://r-lib.github.io/p/pak/devel/'); \
    pak::local_install_deps('/build'); \
    pak::pkg_install('mlverse/cudatoolkit/cuda12.8')"

# Install torch with tests
RUN R CMD INSTALL --install-tests /build

# Trigger LibTorch download
RUN Rscript -e "library(torch)"

ENV TORCH_TEST=1
