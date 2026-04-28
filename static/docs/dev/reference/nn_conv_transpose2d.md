# ConvTranpose2D module

Applies a 2D transposed convolution operator over an input image
composed of several input planes.

## Usage

``` r
nn_conv_transpose2d(
  in_channels,
  out_channels,
  kernel_size,
  stride = 1,
  padding = 0,
  output_padding = 0,
  groups = 1,
  bias = TRUE,
  dilation = 1,
  padding_mode = "zeros"
)
```

## Arguments

- in_channels:

  (int): Number of channels in the input image

- out_channels:

  (int): Number of channels produced by the convolution

- kernel_size:

  (int or tuple): Size of the convolving kernel

- stride:

  (int or tuple, optional): Stride of the convolution. Default: 1

- padding:

  (int or tuple, optional): `dilation * (kernel_size - 1) - padding`
  zero-padding will be added to both sides of each dimension in the
  input. Default: 0

- output_padding:

  (int or tuple, optional): Additional size added to one side of each
  dimension in the output shape. Default: 0

- groups:

  (int, optional): Number of blocked connections from input channels to
  output channels. Default: 1

- bias:

  (bool, optional): If `True`, adds a learnable bias to the output.
  Default: `True`

- dilation:

  (int or tuple, optional): Spacing between kernel elements. Default: 1

- padding_mode:

  (string, optional): `'zeros'`, `'reflect'`, `'replicate'` or
  `'circular'`. Default: `'zeros'`

## Details

This module can be seen as the gradient of Conv2d with respect to its
input. It is also known as a fractionally-strided convolution or a
deconvolution (although it is not an actual deconvolution operation).

- `stride` controls the stride for the cross-correlation.

- `padding` controls the amount of implicit zero-paddings on both sides
  for `dilation * (kernel_size - 1) - padding` number of points. See
  note below for details.

- `output_padding` controls the additional size added to one side of the
  output shape. See note below for details.

- `dilation` controls the spacing between the kernel points; also known
  as the à trous algorithm. It is harder to describe, but this `link`\_
  has a nice visualization of what `dilation` does.

- `groups` controls the connections between inputs and outputs.
  `in_channels` and `out_channels` must both be divisible by `groups`.
  For example,

  - At groups=1, all inputs are convolved to all outputs.

  - At groups=2, the operation becomes equivalent to having two conv
    layers side by side, each seeing half the input channels, and
    producing half the output channels, and both subsequently
    concatenated.

  - At groups= `in_channels`, each input channel is convolved with its
    own set of filters (of size
    \\\left\lfloor\frac{out\\channels}{in\\channels}\right\rfloor\\).

The parameters `kernel_size`, `stride`, `padding`, `output_padding` can
either be:

- a single `int` – in which case the same value is used for the height
  and width dimensions

- a `tuple` of two ints – in which case, the first `int` is used for the
  height dimension, and the second `int` for the width dimension

## Note

Depending of the size of your kernel, several (of the last) columns of
the input might be lost, because it is a valid `cross-correlation`\_,
and not a full `cross-correlation`. It is up to the user to add proper
padding.

The `padding` argument effectively adds
`dilation * (kernel_size - 1) - padding` amount of zero padding to both
sizes of the input. This is set so that when a
[nn_conv2d](https://torch.mlverse.org/docs/dev/reference/nn_conv2d.md)
and a nn_conv_transpose2d are initialized with same parameters, they are
inverses of each other in regard to the input and output shapes.
However, when `stride > 1`,
[nn_conv2d](https://torch.mlverse.org/docs/dev/reference/nn_conv2d.md)
maps multiple input shapes to the same output shape. `output_padding` is
provided to resolve this ambiguity by effectively increasing the
calculated output shape on one side. Note that `output_padding` is only
used to find output shape, but does not actually add zero-padding to
output.

In some circumstances when using the CUDA backend with CuDNN, this
operator may select a nondeterministic algorithm to increase
performance. If this is undesirable, you can try to make the operation
deterministic (potentially at a performance cost) by setting
`torch.backends.cudnn.deterministic = TRUE`.

## Shape

- Input: \\(N, C\_{in}, H\_{in}, W\_{in})\\

- Output: \\(N, C\_{out}, H\_{out}, W\_{out})\\ where \$\$ H\_{out} =
  (H\_{in} - 1) \times \mbox{stride}\[0\] - 2 \times
  \mbox{padding}\[0\] + \mbox{dilation}\[0\] \times
  (\mbox{kernel\\size}\[0\] - 1) + \mbox{output\\padding}\[0\] + 1 \$\$
  \$\$ W\_{out} = (W\_{in} - 1) \times \mbox{stride}\[1\] - 2 \times
  \mbox{padding}\[1\] + \mbox{dilation}\[1\] \times
  (\mbox{kernel\\size}\[1\] - 1) + \mbox{output\\padding}\[1\] + 1 \$\$

## Attributes

- weight (Tensor): the learnable weights of the module of shape
  \\(\mbox{in\\channels}, \frac{\mbox{out\\channels}}{\mbox{groups}},\\
  \\\mbox{kernel\\size\[0\]}, \mbox{kernel\\size\[1\]})\\. The values of
  these weights are sampled from \\\mathcal{U}(-\sqrt{k}, \sqrt{k})\\
  where \\k = \frac{groups}{C\_{\mbox{out}} \*
  \prod\_{i=0}^{1}\mbox{kernel\\size}\[i\]}\\

- bias (Tensor): the learnable bias of the module of shape
  (out_channels) If `bias` is `True`, then the values of these weights
  are sampled from \\\mathcal{U}(-\sqrt{k}, \sqrt{k})\\ where \\k =
  \frac{groups}{C\_{\mbox{out}} \*
  \prod\_{i=0}^{1}\mbox{kernel\\size}\[i\]}\\

## Examples

``` r
if (torch_is_installed()) {
# With square kernels and equal stride
m <- nn_conv_transpose2d(16, 33, 3, stride = 2)
# non-square kernels and unequal stride and with padding
m <- nn_conv_transpose2d(16, 33, c(3, 5), stride = c(2, 1), padding = c(4, 2))
input <- torch_randn(20, 16, 50, 100)
output <- m(input)
# exact output size can be also specified as an argument
input <- torch_randn(1, 16, 12, 12)
downsample <- nn_conv2d(16, 16, 3, stride = 2, padding = 1)
upsample <- nn_conv_transpose2d(16, 16, 3, stride = 2, padding = 1)
h <- downsample(input)
h$size()
output <- upsample(h, output_size = input$size())
output$size()
}
#> [1]  1 16 12 12
```
