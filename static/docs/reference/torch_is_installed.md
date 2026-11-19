# Verifies if torch is installed

Verifies if torch is installed

## Usage

``` r
torch_is_installed(recheck = FALSE)
```

## Arguments

- recheck:

  If `TRUE`, forces rechecking if torch can be loaded in a spearate R
  process. still respects the `TORCH_VERIFY_LOAD` env var.

## Value

TRUE if torch is installed and can be loaded, FALSE otherwise.
