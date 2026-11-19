# Hardtanh

Applies the HardTanh function element-wise.

## Usage

``` r
nnf_hardtanh(input, min_val = -1, max_val = 1, inplace = FALSE)

nnf_hardtanh_(input, min_val = -1, max_val = 1)
```

## Arguments

- input:

  (N,\*) tensor, where \* means, any number of additional dimensions

- min_val:

  minimum value of the linear region range. Default: -1

- max_val:

  maximum value of the linear region range. Default: 1

- inplace:

  can optionally do the operation in-place. Default: FALSE
