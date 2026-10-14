# Adaptive_max_pool3d

Applies a 3D adaptive max pooling over an input signal composed of
several input planes.

## Usage

``` r
nnf_adaptive_max_pool3d(input, output_size, return_indices = FALSE)
```

## Arguments

- input:

  input tensor (minibatch, in_channels , iT \* iH , iW)

- output_size:

  the target output size (single integer or triple-integer tuple)

- return_indices:

  whether to return pooling indices. Default:`FALSE`
