# Rrelu\_

Rrelu\_

## Usage

``` r
torch_rrelu_(
  self,
  lower = 0.125,
  upper = 0.333333333333333,
  training = FALSE,
  generator = NULL
)
```

## Arguments

- self:

  the input tensor

- lower:

  lower bound of the uniform distribution. Default: 1/8

- upper:

  upper bound of the uniform distribution. Default: 1/3

- training:

  bool wether it's a training pass. DEfault: FALSE

- generator:

  random number generator

## rrelu\_(input, lower=1./8, upper=1./3, training=False) -\> Tensor

In-place version of `torch_rrelu`.
