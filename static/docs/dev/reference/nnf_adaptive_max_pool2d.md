# Adaptive_max_pool2d

Applies a 2D adaptive max pooling over an input signal composed of
several input planes.

## Usage

``` r
nnf_adaptive_max_pool2d(input, output_size, return_indices = FALSE)
```

## Arguments

- input:

  input tensor (minibatch, in_channels , iH , iW)

- output_size:

  the target output size (single integer or double-integer tuple)

- return_indices:

  whether to return pooling indices. Default: `FALSE`
