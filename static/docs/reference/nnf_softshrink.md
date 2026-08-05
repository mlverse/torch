# Softshrink

Applies the soft shrinkage function elementwise

## Usage

``` r
nnf_softshrink(input, lambd = 0.5)
```

## Arguments

- input:

  (N,\*) tensor, where \* means, any number of additional dimensions

- lambd:

  the lambda (must be no less than zero) value for the Softshrink
  formulation. Default: 0.5
