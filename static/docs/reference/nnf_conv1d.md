# Conv1d

Applies a 1D convolution over an input signal composed of several input
planes.

## Usage

``` r
nnf_conv1d(
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

  input tensor of shape (minibatch, in_channels , iW)

- weight:

  filters of shape (out_channels, in_channels/groups , kW)

- bias:

  optional bias of shape (out_channels). Default: `NULL`

- stride:

  the stride of the convolving kernel. Can be a single number or a
  one-element tuple `(sW,)`. Default: 1

- padding:

  implicit paddings on both sides of the input. Can be a single number
  or a one-element tuple `(padW,)`. Default: 0

- dilation:

  the spacing between kernel elements. Can be a single number or a
  one-element tuple `(dW,)`. Default: 1

- groups:

  split input into groups, `in_channels` should be divisible by the
  number of groups. Default: 1
