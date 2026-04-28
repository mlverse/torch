# Avg_pool3d

Applies 3D average-pooling operation in \\kT \* kH \* kW\\ regions by
step size \\sT \* sH \* sW\\ steps. The number of output features is
equal to \\\lfloor \frac{ \mbox{input planes} }{sT} \rfloor\\.

## Usage

``` r
nnf_avg_pool3d(
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

  input tensor (minibatch, in_channels , iT \* iH , iW)

- kernel_size:

  size of the pooling region. Can be a single number or a tuple
  `(kT, kH, kW)`

- stride:

  stride of the pooling operation. Can be a single number or a tuple
  `(sT, sH, sW)`. Default: `kernel_size`

- padding:

  implicit zero paddings on both sides of the input. Can be a single
  number or a tuple `(padT, padH, padW)`, Default: 0

- ceil_mode:

  when True, will use `ceil` instead of `floor` in the formula to
  compute the output shape

- count_include_pad:

  when True, will include the zero-padding in the averaging calculation

- divisor_override:

  NA if specified, it will be used as divisor, otherwise size of the
  pooling region will be used. Default: `NULL`
