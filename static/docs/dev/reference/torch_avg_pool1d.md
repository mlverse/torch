# Avg_pool1d

Avg_pool1d

## Usage

``` r
torch_avg_pool1d(
  self,
  kernel_size,
  stride = list(),
  padding = 0L,
  ceil_mode = FALSE,
  count_include_pad = TRUE
)
```

## Arguments

- self:

  input tensor of shape \\(\mbox{minibatch} , \mbox{in\\channels} ,
  iW)\\

- kernel_size:

  the size of the window. Can be a single number or a tuple `(kW,)`

- stride:

  the stride of the window. Can be a single number or a tuple `(sW,)`.
  Default: `kernel_size`

- padding:

  implicit zero paddings on both sides of the input. Can be a single
  number or a tuple `(padW,)`. Default: 0

- ceil_mode:

  when `TRUE`, will use `ceil` instead of `floor` to compute the output
  shape. Default: `FALSE`

- count_include_pad:

  when `TRUE`, will include the zero-padding in the averaging
  calculation. Default: `TRUE`

## avg_pool1d(input, kernel_size, stride=NULL, padding=0, ceil_mode=FALSE, count_include_pad=TRUE) -\> Tensor

Applies a 1D average pooling over an input signal composed of several
input planes.

See
[`nn_avg_pool1d()`](https://torch.mlverse.org/docs/dev/reference/nn_avg_pool1d.md)
for details and output shape.
