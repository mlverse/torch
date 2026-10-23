# Conv_transpose1d

Applies a 1D transposed convolution operator over an input signal
composed of several input planes, sometimes also called "deconvolution".

## Usage

``` r
nnf_conv_transpose1d(
  input,
  weight,
  bias = NULL,
  stride = 1,
  padding = 0,
  output_padding = 0,
  groups = 1,
  dilation = 1
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

- output_padding:

  padding applied to the output

- groups:

  split input into groups, `in_channels` should be divisible by the
  number of groups. Default: 1

- dilation:

  the spacing between kernel elements. Can be a single number or a
  one-element tuple `(dW,)`. Default: 1
