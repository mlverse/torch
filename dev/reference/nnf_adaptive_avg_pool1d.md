# Adaptive_avg_pool1d

Applies a 1D adaptive average pooling over an input signal composed of
several input planes.

## Usage

``` r
nnf_adaptive_avg_pool1d(input, output_size)
```

## Arguments

- input:

  input tensor of shape (minibatch , in_channels , iW)

- output_size:

  the target output size (single integer)
