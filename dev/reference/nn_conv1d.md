# Conv1D module

Applies a 1D convolution over an input signal composed of several input
planes. In the simplest case, the output value of the layer with input
size \\(N, C\_{\mbox{in}}, L)\\ and output \\(N, C\_{\mbox{out}},
L\_{\mbox{out}})\\ can be precisely described as:

## Usage

``` r
nn_conv1d(
  in_channels,
  out_channels,
  kernel_size,
  stride = 1,
  padding = 0,
  dilation = 1,
  groups = 1,
  bias = TRUE,
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

  (int, tuple or str, optional) – Padding added to both sides of the
  input. Default: 0

- dilation:

  (int or tuple, optional): Spacing between kernel elements. Default: 1

- groups:

  (int, optional): Number of blocked connections from input channels to
  output channels. Default: 1

- bias:

  (bool, optional): If `TRUE`, adds a learnable bias to the output.
  Default: `TRUE`

- padding_mode:

  (string, optional): `'zeros'`, `'reflect'`, `'replicate'` or
  `'circular'`. Default: `'zeros'`

## Details

\$\$ \mbox{out}(N_i, C\_{\mbox{out}\_j}) =
\mbox{bias}(C\_{\mbox{out}\_j}) + \sum\_{k = 0}^{C\_{in} - 1}
\mbox{weight}(C\_{\mbox{out}\_j}, k) \star \mbox{input}(N_i, k) \$\$

where \\\star\\ is the valid cross correlation operator, \\N\\ is a
batch size, \\C\\ denotes a number of channels, \\L\\ is a length of
signal sequence.

- `stride` controls the stride for the cross-correlation, a single
  number or a one-element tuple.

- `padding` controls the amount of implicit zero-paddings on both sides
  for `padding` number of points.

- `dilation` controls the spacing between the kernel points; also known
  as the à trous algorithm. It is harder to describe, but this
  [link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)
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
    own set of filters, of size
    \\\left\lfloor\frac{out\\channels}{in\\channels}\right\rfloor\\.

## Note

Depending of the size of your kernel, several (of the last) columns of
the input might be lost, because it is a valid `cross-correlation`*, and
not a full `cross-correlation`*. It is up to the user to add proper
padding.

When `groups == in_channels` and `out_channels == K * in_channels`,
where `K` is a positive integer, this operation is also termed in
literature as depthwise convolution. In other words, for an input of
size \\(N, C\_{in}, L\_{in})\\, a depthwise convolution with a depthwise
multiplier `K`, can be constructed by arguments
\\(C\_{\mbox{in}}=C\_{in}, C\_{\mbox{out}}=C\_{in} \times K, ...,
\mbox{groups}=C\_{in})\\.

## Shape

- Input: \\(N, C\_{in}, L\_{in})\\

- Output: \\(N, C\_{out}, L\_{out})\\ where

\$\$ L\_{out} = \left\lfloor\frac{L\_{in} + 2 \times \mbox{padding} -
\mbox{dilation} \times (\mbox{kernel\\size} - 1) - 1}{\mbox{stride}} +
1\right\rfloor \$\$

## Attributes

- weight (Tensor): the learnable weights of the module of shape
  \\(\mbox{out\\channels}, \frac{\mbox{in\\channels}}{\mbox{groups}},
  \mbox{kernel\\size})\\. The values of these weights are sampled from
  \\\mathcal{U}(-\sqrt{k}, \sqrt{k})\\ where \\k =
  \frac{groups}{C\_{\mbox{in}} \* \mbox{kernel\\size}}\\

- bias (Tensor): the learnable bias of the module of shape
  (out_channels). If `bias` is `TRUE`, then the values of these weights
  are sampled from \\\mathcal{U}(-\sqrt{k}, \sqrt{k})\\ where \\k =
  \frac{groups}{C\_{\mbox{in}} \* \mbox{kernel\\size}}\\

## Examples

``` r
if (torch_is_installed()) {
m <- nn_conv1d(16, 33, 3, stride = 2)
input <- torch_randn(20, 16, 50)
output <- m(input)
}
```
