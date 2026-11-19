# Gelu

Gelu

## Usage

``` r
nnf_gelu(input, approximate = "none")
```

## Arguments

- input:

  (N,\*) tensor, where \* means, any number of additional dimensions

- approximate:

  By default it's none, and applies element-wise x\*pnorm(x), if 'tanh',
  then GELU is estimated. See [GELU](https://arxiv.org/abs/1606.08415)
  for more info.

## gelu(input) -\> Tensor

Applies element-wise the function \\GELU(x) = x \* \Phi(x)\\

where \\\Phi(x)\\ is the Cumulative Distribution Function for Gaussian
Distribution.

See [Gaussian Error Linear Units
(GELUs)](https://arxiv.org/abs/1606.08415).
