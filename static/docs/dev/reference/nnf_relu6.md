# Relu6

Applies the element-wise function \\ReLU6(x) = min(max(0,x), 6)\\.

## Usage

``` r
nnf_relu6(input, inplace = FALSE)
```

## Arguments

- input:

  (N,\*) tensor, where \* means, any number of additional dimensions

- inplace:

  can optionally do the operation in-place. Default: FALSE
