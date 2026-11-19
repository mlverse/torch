# Install Torch from files

List the Torch and Lantern libraries URLs to download as local files in
order to proceed with `install_torch_from_file()`.

Installs Torch and its dependencies from files.

## Usage

``` r
get_install_libs_url(version = NA, type = NA)

install_torch_from_file(version = NA, type = NA, libtorch, liblantern, ...)
```

## Arguments

- version:

  Not used

- type:

  Not used. This function is deprecated.

- libtorch:

  The installation archive file to use for Torch. Shall be a `"file://"`
  URL scheme.

- liblantern:

  The installation archive file to use for Lantern. Shall be a
  `"file://"` URL scheme.

- ...:

  other parameters to be passed to `"install_torch()"`

## Details

When `"install_torch()"` initiated download is not possible, but
installation archive files are present on local filesystem,
`"install_torch_from_file()"` can be used as a workaround to
installation issue. `"libtorch"` is the archive containing all torch
modules, and `"liblantern"` is the C interface to libtorch that is used
for the R package. Both are highly dependent, and should be checked
through `"get_install_libs_url()"`

## Examples

``` r
if (torch_is_installed()) {
if (FALSE) { # \dontrun{
# on a linux CPU platform 
get_install_libs_url()
# then after making both files available into /tmp/
Sys.setenv(TORCH_URL="/tmp/libtorch-v1.13.1.zip")
Sys.setenv(LANTERN_URL="/tmp/lantern-0.9.1.9001+cpu+arm64-Darwin.zip")
torch::install_torch()
} # }
}
```
