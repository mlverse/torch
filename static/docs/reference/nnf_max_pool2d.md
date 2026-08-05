# Max_pool2d

Applies a 2D max pooling over an input signal composed of several input
planes.

## Usage

``` r
nnf_max_pool2d(
  input,
  kernel_size,
  stride = kernel_size,
  padding = 0,
  dilation = 1,
  ceil_mode = FALSE,
  return_indices = FALSE
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

- dilation:

  controls the spacing between the kernel points; also known as the à
  trous algorithm.

- ceil_mode:

  when True, will use `ceil` instead of `floor` in the formula to
  compute the output shape. Default: `FALSE`

- return_indices:

  whether to return the indices where the max occurs.
