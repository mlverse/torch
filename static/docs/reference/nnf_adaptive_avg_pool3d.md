# Adaptive_avg_pool3d

Applies a 3D adaptive average pooling over an input signal composed of
several input planes.

## Usage

``` r
nnf_adaptive_avg_pool3d(input, output_size)
```

## Arguments

- input:

  input tensor (minibatch, in_channels , iT \* iH , iW)

- output_size:

  the target output size (single integer or triple-integer tuple)
