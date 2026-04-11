# Install Torch

Installs Torch and its dependencies.

## Usage

``` r
install_torch(reinstall = FALSE, ..., .inform_restart = TRUE)
```

## Arguments

- reinstall:

  Re-install Torch even if its already installed?

- ...:

  Currently unused.

- .inform_restart:

  if `TRUE` and running in an
  [`interactive()`](https://rdrr.io/r/base/interactive.html) session,
  after installation it will print a message to inform the user that the
  session must be restarted for torch to work correctly.

## Details

This function is mainly controlled by environment variables that can be
used to override the defaults:

- `TORCH_HOME`: the installation path. By default dependencies are
  installed within the package directory. Eg what's given by
  `system.file(package="torch")`.

- `TORCH_URL`: A URL, path to a ZIP file or a directory containing a
  LibTorch version. Files will be installed/copied to the `TORCH_HOME`
  directory.

- `LANTERN_URL`: Same as `TORCH_URL` but for the Lantern library.

- `TORCH_INSTALL_DEBUG`: Setting it to 1, shows debug log messages
  during installation.

- `PRECXX11ABI`: DEPRECATED. No longer has effects. Setting it to `1`
  will will trigger the installation of a Pre-cxx11 ABI installation of
  LibTorch. This can be useful in environments with older versions of
  GLIBC like CentOS7 and older Debian/Ubuntu versions.

- `LANTERN_BASE_URL`: The base URL for lantern files. This allows
  passing a directory where lantern binaries are located. The filename
  is then constructed as usual.

- `TORCH_COMMIT_SHA`: torch repository commit sha to be used when
  querying lantern uploads. Set it to `'none'` to avoid looking for
  build for that commit and use the latest build for the branch.

- `CUDA`: We try to automatically detect the CUDA version installed in
  your system, but you might want to manually set it here. You can also
  disable CUDA installation by setting it to 'cpu'.

- `TORCH_R_VERSION`: The R torch version. It's unlikely that you need to
  change it, but it can be useful if you don't have the R package
  installed, but want to install the dependencies.

The `TORCH_INSTALL` environment variable can be set to `0` to prevent
auto-installing torch and `TORCH_LOAD` set to `0` to avoid loading
dependencies automatically. These environment variables are meant for
advanced use cases and troubleshooting only. When timeout error occurs
during library archive download, or length of downloaded files differ
from reported length, an increase of the `timeout` value should help.
