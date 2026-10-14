# Prelu

Applies element-wise the function \\PReLU(x) = max(0,x) + weight \*
min(0,x)\\ where weight is a learnable parameter.

## Usage

``` r
nnf_prelu(input, weight)
```

## Arguments

- input:

  (N,\*) tensor, where \* means, any number of additional dimensions

- weight:

  (Tensor) the learnable weights
