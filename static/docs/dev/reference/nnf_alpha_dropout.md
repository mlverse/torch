# Alpha_dropout

Applies alpha dropout to the input.

## Usage

``` r
nnf_alpha_dropout(input, p = 0.5, training = FALSE, inplace = FALSE)
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
