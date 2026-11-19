# Celu

Applies element-wise, \\CELU(x) = max(0,x) + min(0, \alpha \* (exp(x
\alpha) - 1))\\.

## Usage

``` r
nnf_celu(input, alpha = 1, inplace = FALSE)

nnf_celu_(input, alpha = 1)
```

## Arguments

- input:

  (N,\*) tensor, where \* means, any number of additional dimensions

- alpha:

  the alpha value for the CELU formulation. Default: 1.0

- inplace:

  can optionally do the operation in-place. Default: FALSE
