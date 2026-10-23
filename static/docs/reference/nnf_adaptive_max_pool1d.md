# Adaptive_max_pool1d

Applies a 1D adaptive max pooling over an input signal composed of
several input planes.

## Usage

``` r
nnf_adaptive_max_pool1d(input, output_size, return_indices = FALSE)
```

## Arguments

- input:

  input tensor of shape (minibatch , in_channels , iW)

- output_size:

  the target output size (single integer)

- return_indices:

  whether to return pooling indices. Default: `FALSE`
