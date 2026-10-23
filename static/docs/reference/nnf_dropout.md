# Dropout

During training, randomly zeroes some of the elements of the input
tensor with probability `p` using samples from a Bernoulli distribution.

## Usage

``` r
nnf_dropout(input, p = 0.5, training = TRUE, inplace = FALSE)
```

## Arguments

- input:

  the input tensor

- p:

  probability of an element to be zeroed. Default: 0.5

- training:

  apply dropout if is `TRUE`. Default: `TRUE`

- inplace:

  If set to `TRUE`, will do this operation in-place. Default: `FALSE`
