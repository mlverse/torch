# Torch Installation Situation Report

Generate a comprehensive diagnostic report for torch installation
status. This function dumps everything relevant to installation and
setup in one go.

## Usage

``` r
install_torch_sitrep(verbose = TRUE)
```

## Arguments

- verbose:

  logical; if TRUE, prints detailed information to console. Default
  TRUE.

## Value

Invisibly returns a list containing all diagnostic information.

## Examples

``` r
if (torch_is_installed()) {
if (FALSE) { # \dontrun{
install_torch_sitrep()
} # }
}
```
