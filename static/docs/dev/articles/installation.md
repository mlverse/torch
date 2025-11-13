# Installation

After the usual R package installation, `torch` requires installing
other 2 libraries: LibTorch and LibLantern. They are automatically
installed by detecting information about you OS if you are using `torch`
in interactive mode. If you are running `torch` in non-interactive
environments you need to set the `TORCH_INSTALL` env var to 1, so it’s
automatically installed or manually call
[`torch::install_torch()`](https://torch.mlverse.org/docs/dev/reference/install_torch.md).

Starting from torch v0.9.1.9000 it’s possible to install torch from
pre-built package binaries available in a custom CRAN-like repository.
This binaries include all shared objects necessary to run torch, thus
it’s not required to install any additional software. See the [*Install
from pre-built binaries*](#pre-built) session for more information.

We have provide pre-compiled binaries for all major platforms and you
can find specific installation instructions below.

## Windows

### CPU

If you don’t have a GPU or want to install the CPU version of `torch`,
you can install with:

``` r
install.packages("torch")
```

Some Windows distributions don’t have the Visual Studio runtime
pre-installed and you will observe an error like:

    Error in cpp_lantern_init(normalizePath(install_path())): C:\Users\User\Documents\R\R-4.0.2\library\torch\deps\lantern.dll - The specified module could not be found.

See
[here](https://github.com/mlverse/torch/issues/246#issuecomment-695097121)
for instructions on how to install it.

### GPU

torch has very specific requirements in terms of CUDA and CUDNN versions
it supports. We recomment installing torch using pre-built binaries. See
(#pre-built) for more information.

Since version 0.1.1 torch supports GPU installation on Windows. In order
to use GPU’s with torch you need to:

- Have a CUDA compatible NVIDIA GPU. You can find if you have a CUDA
  compatible GPU [here](https://developer.nvidia.com/cuda-gpus#compute).

- Have properly installed the NVIDIA CUDA toolkit version 12.8. For CUDA
  v12.8, follow the installation instructions
  [here](https://developer.nvidia.com/cuda-12-8-1-download-archive).
  **Note**: The version of the CUDA toolkit must match exactly what’s
  mentioed above.

- Have installed cuDNN - a version compatible with CUDA v12.8. Follow
  the installation instructions available
  [here](https://developer.nvidia.com/cudnn).

Once you have installed all pre-requisites you can install `torch` with:

``` r
install.packages("torch")
```

If you have followed default installation locations we will detect that
you have CUDA software installed and automatically download the GPU
enabled Lantern binaries. You can also specify the `CUDA` env var with
something like `Sys.setenv(CUDA="11.7")` if you want to force an
specific version of the CUDA toolkit.

## MacOS

### CPU

We support CPU builds of torch on MacOS. On MacOS you can install torch
with:

``` r
install.packages("torch")
```

### GPU

On Apple Silicon architecture we support GPU through MPS:

``` r
install.packages("torch")
```

## Linux

### CPU

To install the cpu version of `torch` you can run:

``` r
install.packages("torch")
```

### GPU

torch has very specific requirements in terms of CUDA and CUDNN versions
it supports. We recomment installing torch using pre-built binaries. See
(#pre-built) for more information.

To install the GPU version of `torch` on linux you must verify that:

- You have a NVIDIA CUDA compatible GPU. You can find if you have a CUDA
  compatible GPU [here](https://developer.nvidia.com/cuda-gpus#compute).

- You have correctly installed the NVIDIA CUDA Toolkit versions 11.6 or
  11.7, follow the instructions
  [here](https://docs.nvidia.com/cuda/archive/11.7.0/).

- You have installed cuDNN (a version compatible with your CUDA
  version). Follow the installation instructions available
  [here](https://developer.nvidia.com/cudnn).

Once you have installed all pre-requisites you can install `torch` with:

``` r
install.packages("torch")
```

If you have followed default installation locations we will detect that
you have CUDA software installed and automatically download the GPU
enabled Lantern binaries. You can also specify the `CUDA` env var with
something like `Sys.setenv(CUDA="12.8")` if you want to force an
specific version of the CUDA toolkit.

## Installing from pre-built binaries

As of torch v0.9.1.9000 it’s now possible to install torch from
pre-built package binaries from a CRAN like repository hosted on Google
Cloud Storage. We currently provide pre-built binaries for CPU (for
macOS, Linux and Windows) and GPU (Windows and Linux).

Packages provided by the CRAN-like repository bundles all necessary for
its execution, including CUDA and CUDNN in the case of the GPU builds.
This means that by installing it **you agree** with the included
software licenses. See PyTorch’s
[LICENSE](https://github.com/pytorch/pytorch/blob/master/LICENSE) and
[CUDA libraries EULA](https://docs.nvidia.com/cuda/eula/).

When installing from the pre-built binaries, you **don’t need** to
manually install CUDA or cuDNN. If you have CUDA installed, it doesn’t
need to match the installation *‘kind’* chosen below.

To install from the pre-built binaries, you can use the following:

``` r
options(timeout = 600) # increasing timeout is recommended since we will be downloading a 2GB file.
# For Windows and Linux: "cpu", "cu128" are the only currently supported
# For MacOS the supported are: "cpu-intel" or "cpu-m1"
kind <- "cu128"
version <- available.packages()["torch","Version"]
options(repos = c(
  torch = sprintf("https://torch-cdn.mlverse.org/packages/%s/%s/", kind, version),
  CRAN = "https://cloud.r-project.org" # or any other from which you want to install the other R dependencies.
))
install.packages("torch")
```

## Troubleshooting

### Large file download timeout

If you encounter timeout during library download, or if after a while,
downloads end-up with a warning such as:

    Warning messages:
    1: In utils::download.file(library_url, temp_file) :
      downloaded length 44901568 != reported length 141774525
    2: In utils::download.file(library_url, temp_file) :
      URL '...': Timeout of 60 seconds was reached
    3: Failed to install Torch, manually run install_torch(). download from 'https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.7.1.zip' failed

This means you encounter a download timeout. then, you should increase
the timeout value in
[`install_torch()`](https://torch.mlverse.org/docs/dev/reference/install_torch.md)
like

``` r
install_torch(timeout = 600)
```

### File based download

In cases where you cannot reach download servers from the machine you
intend to install torch on, last resort is to install Torch and Lantern
library from files. This is done in 3 steps :

1- get the download URLs of the files.

``` r
get_install_libs_url()
```

2- save those files into the machine filesystem. We will use `/tmp/`
here as an example .

3- install torch from files

``` r
# then after making both files available into /tmp/
Sys.setenv(TORCH_URL="/tmp/libtorch-v1.13.1.zip")
Sys.setenv(LANTERN_URL="/tmp/lantern-0.9.1.9001+cpu+arm64-Darwin.zip")
torch::install_torch()
```

### Installing older versions

As of v0.13.0 torch shifted from using Google Cloud Storage service to
AWS S3 as the storage service for the required Lantern binaries.

We will keep the files in the GCS bucket for as long as possible, but we
might need to remove them at some point in time. Those files have been
backed up in the new AWS S3 bucket using the same file structure, so if
torch tries to download some from a URL starting with
`https://storage.googleapis.com/torch-lantern-builds`, you should now
replace it with `https://torch-cdn.mlverse.org`.

For torch versions between v0.10.0 and v0.12.0 (both included), you
should be able to set the environment variable
`LANTERN_BASE_URL=https://torch-cdn.mlverse.org/binaries/` to point to
the new address of the binaries. For older versions of torch, you might
need to manually download the file from the new address and extract it
to the expected `TORCH_HOME` directory. Feel free to open an issue on
GitHub if you need help with this.
