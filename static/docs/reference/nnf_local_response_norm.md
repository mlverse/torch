# Local_response_norm

Applies local response normalization over an input signal composed of
several input planes, where channels occupy the second dimension.
Applies normalization across channels.

## Usage

``` r
nnf_local_response_norm(input, size, alpha = 1e-04, beta = 0.75, k = 1)
```

## Arguments

- input:

  the input tensor

- size:

  amount of neighbouring channels used for normalization

- alpha:

  multiplicative factor. Default: 0.0001

- beta:

  exponent. Default: 0.75

- k:

  additive factor. Default: 1
