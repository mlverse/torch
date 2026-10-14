# Avg_pool2d

Applies 2D average-pooling operation in \\kH \* kW\\ regions by step
size \\sH \* sW\\ steps. The number of output features is equal to the
number of input planes.

## Usage

``` r
nnf_avg_pool2d(
  input,
  kernel_size,
  stride = NULL,
  padding = 0,
  ceil_mode = FALSE,
  count_include_pad = TRUE,
  divisor_override = NULL
)
```

## Arguments

- input:

  input tensor (minibatch, in_channels , iH , iW)

- kernel_size:

  size of the pooling region. Can be a single number or a tuple
  `(kH, kW)`

- stride:

  stride of the pooling operation. Can be a single number or a tuple
  `(sH, sW)`. Default: `kernel_size`

- padding:

  implicit zero paddings on both sides of the input. Can be a single
  number or a tuple `(padH, padW)`. Default: 0

- ceil_mode:

  when True, will use `ceil` instead of `floor` in the formula to
  compute the output shape. Default: `FALSE`

- count_include_pad:

  when True, will include the zero-padding in the averaging calculation.
  Default: `TRUE`

- divisor_override:

  if specified, it will be used as divisor, otherwise size of the
  pooling region will be used. Default: `NULL`
