# Max_pool3d

Applies a 3D max pooling over an input signal composed of several input
planes.

## Usage

``` r
nnf_max_pool3d(
  input,
  kernel_size,
  stride = NULL,
  padding = 0,
  dilation = 1,
  ceil_mode = FALSE,
  return_indices = FALSE
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

- dilation:

  controls the spacing between the kernel points; also known as the à
  trous algorithm.

- ceil_mode:

  when True, will use `ceil` instead of `floor` in the formula to
  compute the output shape

- return_indices:

  whether to return the indices where the max occurs.
