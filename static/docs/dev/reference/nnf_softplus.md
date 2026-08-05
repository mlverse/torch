# Softplus

Applies element-wise, the function \\Softplus(x) = 1/\beta \* log(1 +
exp(\beta \* x))\\.

## Usage

``` r
nnf_softplus(input, beta = 1, threshold = 20)
```

## Arguments

- input:

  (N,\*) tensor, where \* means, any number of additional dimensions

- beta:

  the beta value for the Softplus formulation. Default: 1

- threshold:

  values above this revert to a linear function. Default: 20

## Details

For numerical stability the implementation reverts to the linear
function when \\input \* \beta \> threshold\\.
