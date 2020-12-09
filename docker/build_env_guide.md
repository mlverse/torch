# Build Torch RStudio with CUDA on Docker
This document is a environmental constrauction guide for Torch RStudio with CUDA on Docker.

## Requirements
Following requirements are expected to work on a Linux machine.
* Docker
* Docker Compose
* NVIDIA Docker 2
* CUDA arch based GPU env

## Building env
### 1. Clone repository
Just doing following command.

```bash
git clone https://github.com/mlverse/torch.git
```

### 2. Change directory
Move into `torch/docker` directory.

```bash
cd torch/docker
```

### 3. Create `rstudio_home` directory
Create `rstudio_home` directory outside of torch directory which uses home directory of container.(just doing following command.) If you would like to use another directory as home directory, you must modify `docker-compose.yml`(`Path` strings at `volume` section).

```bash
mkdir  ../../rstudio_home
```

### 4. Create `auth.txt`
Create `auth.txt` into `torch/docker/` and write root user's pass word and rstudio(default user) user's password which you would like to set.

```bash
echo "rstudio" >> auth.txt # root user's pass word
echo "rstudio" >> ahth.txt # rstudio user's pass word
```

### 5. Run `build_env.sh`
You just implement `build_env.sh`, and you can build a Torch RStudio container image.

```bash
./build_env.sh
```

If you would like to build a container image again without using chache files, you should use `-nc` option.

```bash
./build_env.sh -nc
```

#### (Option): Building Japanese env
If you would like to use Japanese version of Torch RStudio env, you should delete following comment outs in Dockerfile.(Line160 to 165)

```Dockerfile
# ENV LANG ja_JP.UTF-8
# ENV LC_ALL ja_JP.UTF-8
# RUN sed -i '$d' /etc/locale.gen && echo "ja_JP.UTF-8 UTF-8" >> /etc/locale.gen \
#   && locale-gen ja_JP.UTF-8 && /usr/sbin/update-locale LANG=ja_JP.UTF-8 LANGUAGE="ja_JP:ja"
# RUN /bin/bash -c "source /etc/default/locale"
# RUN ln -sf  /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
```

### 6. Start Torch RStudio Container
Just doing following command.

```bash
docker-compose up -d
```

### 7. Login RStudio
Open your web browser and access `localhost:8787`.

You implement `torch` first time, additional pacckages which needs for it will be installed in your env. If your `GPU` is available, `cuda_is_available()` returns `True`.

### 8. Delete `auth.txt`
If you would like not to leak out your container's pass word, delete `auth.txt`.

```bash
rm auth.txt
```