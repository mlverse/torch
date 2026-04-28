# ConvTranspose1D

Applies a 1D transposed convolution operator over an input image
composed of several input planes.

## Usage

``` r
nn_conv_transpose1d(
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
  zero-padding will be added to both sides of the input. Default: 0

- output_padding:

  (int or tuple, optional): Additional size added to one side of the
  output shape. Default: 0

- groups:

  (int, optional): Number of blocked connections from input channels to
  output channels. Default: 1

- bias:

  (bool, optional): If `True`, adds a learnable bias to the output.
  Default: `TRUE`

- dilation:

  (int or tuple, optional): Spacing between kernel elements. Default: 1

- padding_mode:

  (string, optional): `'zeros'`, `'reflect'`, `'replicate'` or
  `'circular'`. Default: `'zeros'`

## Details

This module can be seen as the gradient of Conv1d with respect to its
input. It is also known as a fractionally-strided convolution or a
deconvolution (although it is not an actual deconvolution operation).

- `stride` controls the stride for the cross-correlation.

- `padding` controls the amount of implicit zero-paddings on both sides
  for `dilation * (kernel_size - 1) - padding` number of points. See
  note below for details.

- `output_padding` controls the additional size added to one side of the
  output shape. See note below for details.

- `dilation` controls the spacing between the kernel points; also known
  as the à trous algorithm. It is harder to describe, but this
  [link](https://github.com/vdumoulin/conv_arithmetic) has a nice
  visualization of what `dilation` does.

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

## Note

Depending of the size of your kernel, several (of the last) columns of
the input might be lost, because it is a valid `cross-correlation`*, and
not a full `cross-correlation`*. It is up to the user to add proper
padding.

The `padding` argument effectively adds
`dilation * (kernel_size - 1) - padding` amount of zero padding to both
sizes of the input. This is set so that when a `~torch.nn.Conv1d` and a
`~torch.nn.ConvTranspose1d` are initialized with same parameters, they
are inverses of each other in regard to the input and output shapes.
However, when `stride > 1`, `~torch.nn.Conv1d` maps multiple input
shapes to the same output shape. `output_padding` is provided to resolve
this ambiguity by effectively increasing the calculated output shape on
one side. Note that `output_padding` is only used to find output shape,
but does not actually add zero-padding to output.

In some circumstances when using the CUDA backend with CuDNN, this
operator may select a nondeterministic algorithm to increase
performance. If this is undesirable, you can try to make the operation
deterministic (potentially at a performance cost) by setting
`torch.backends.cudnn.deterministic = TRUE`.

## Shape

- Input: \\(N, C\_{in}, L\_{in})\\

- Output: \\(N, C\_{out}, L\_{out})\\ where \$\$ L\_{out} =
  (L\_{in} - 1) \times \mbox{stride} - 2 \times \mbox{padding} +
  \mbox{dilation} \times (\mbox{kernel\\size} - 1) +
  \mbox{output\\padding} + 1 \$\$

## Attributes

- weight (Tensor): the learnable weights of the module of shape
  \\(\mbox{in\\channels}, \frac{\mbox{out\\channels}}{\mbox{groups}},\\
  \\\mbox{kernel\\size})\\. The values of these weights are sampled from
  \\\mathcal{U}(-\sqrt{k}, \sqrt{k})\\ where \\k =
  \frac{groups}{C\_{\mbox{out}} \* \mbox{kernel\\size}}\\

- bias (Tensor): the learnable bias of the module of shape
  (out_channels). If `bias` is `TRUE`, then the values of these weights
  are sampled from \\\mathcal{U}(-\sqrt{k}, \sqrt{k})\\ where \\k =
  \frac{groups}{C\_{\mbox{out}} \* \mbox{kernel\\size}}\\

## Examples

``` r
if (torch_is_installed()) {
m <- nn_conv_transpose1d(32, 16, 2)
input <- torch_randn(10, 32, 2)
output <- m(input)
}
```
