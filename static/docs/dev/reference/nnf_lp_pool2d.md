# Lp_pool2d

Applies a 2D power-average pooling over an input signal composed of
several input planes. If the sum of all inputs to the power of `p` is
zero, the gradient is set to zero as well.

## Usage

``` r
nnf_lp_pool2d(input, norm_type, kernel_size, stride = NULL, ceil_mode = FALSE)
```

## Arguments

- input:

  the input tensor

- norm_type:

  if inf than one gets max pooling if 0 you get sum pooling (
  proportional to the avg pooling)

- kernel_size:

  a single int, the size of the window

- stride:

  a single int, the stride of the window. Default value is kernel_size

- ceil_mode:

  when True, will use ceil instead of floor to compute the output shape
