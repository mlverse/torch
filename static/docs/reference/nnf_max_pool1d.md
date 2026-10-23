# Max_pool1d

Applies a 1D max pooling over an input signal composed of several input
planes.

## Usage

``` r
nnf_max_pool1d(
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

  input tensor of shape (minibatch , in_channels , iW)

- kernel_size:

  the size of the window. Can be a single number or a tuple `(kW,)`.

- stride:

  the stride of the window. Can be a single number or a tuple `(sW,)`.
  Default: `kernel_size`

- padding:

  implicit zero paddings on both sides of the input. Can be a single
  number or a tuple `(padW,)`. Default: 0

- dilation:

  controls the spacing between the kernel points; also known as the à
  trous algorithm.

- ceil_mode:

  when True, will use `ceil` instead of `floor` to compute the output
  shape. Default: `FALSE`

- return_indices:

  whether to return the indices where the max occurs.
