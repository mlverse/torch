FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update -y && apt-get install -y \
    sudo software-properties-common dialog apt-utils \
    tzdata locales curl wget git \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    libfontconfig1-dev libfreetype6-dev libpng-dev \
    libharfbuzz-dev libfribidi-dev libtiff5-dev libjpeg-dev \
    cmake make gcc g++

# Install R via rig
RUN curl -L https://rig.r-pkg.org/deb/rig.gpg -o /etc/apt/trusted.gpg.d/rig.gpg \
    && echo "deb http://rig.r-pkg.org/deb rig main" > /etc/apt/sources.list.d/rig.list \
    && apt-get update \
    && apt-get install -y r-rig \
    && rig add release \
    && rig default release \
    && rm -rf /var/lib/apt/lists/*

# Use a fixed library path (not HOME-dependent) so packages are found
# regardless of what HOME is set to at runtime (GitHub Actions sets HOME=/github/home)
ENV R_LIBS_USER=/opt/R/library
RUN mkdir -p /opt/R/library

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
    pak::local_install_deps('/build', dependencies = TRUE); \
    pak::pkg_install('mlverse/cudatoolkit/cuda12.8')" \
    && rm -rf /tmp/* /root/.cache

# Install torch with tests
RUN R CMD INSTALL --install-tests /build

# Trigger LibTorch download and clean up temp files
RUN Rscript -e "library(torch)" \
    && rm -rf /tmp/* /build

ENV TORCH_TEST=1
