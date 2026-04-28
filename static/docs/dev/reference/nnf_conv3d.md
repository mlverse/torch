# Conv3d

Applies a 3D convolution over an input image composed of several input
planes.

## Usage

``` r
nnf_conv3d(
  input,
  weight,
  bias = NULL,
  stride = 1,
  padding = 0,
  dilation = 1,
  groups = 1
)
```

## Arguments

- input:

  input tensor of shape (minibatch, in_channels , iT , iH , iW)

- weight:

  filters of shape (out_channels , in_channels/groups, kT , kH , kW)

- bias:

  optional bias tensor of shape (out_channels). Default: `NULL`

- stride:

  the stride of the convolving kernel. Can be a single number or a tuple
  `(sT, sH, sW)`. Default: 1

- padding:

  implicit paddings on both sides of the input. Can be a single number
  or a tuple `(padT, padH, padW)`. Default: 0

- dilation:

  the spacing between kernel elements. Can be a single number or a tuple
  `(dT, dH, dW)`. Default: 1

- groups:

  split input into groups, `in_channels` should be divisible by the
  number of groups. Default: 1
