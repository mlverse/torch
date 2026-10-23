# Threshold

Thresholds each element of the input Tensor.

## Usage

``` r
nnf_threshold(input, threshold, value, inplace = FALSE)

nnf_threshold_(input, threshold, value)
```

## Arguments

- input:

  (N,\*) tensor, where \* means, any number of additional dimensions

- threshold:

  The value to threshold at

- value:

  The value to replace with

- inplace:

  can optionally do the operation in-place. Default: FALSE
