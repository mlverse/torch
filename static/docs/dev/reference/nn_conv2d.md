# Conv2D module

Applies a 2D convolution over an input signal composed of several input
planes.

## Usage

``` r
nn_conv2d(
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

  (int or tuple or string, optional): Zero-padding added to both sides
  of the input. controls the amount of padding applied to the input. It
  can be either a string `'valid'`, `'same'` or a tuple of ints giving
  the amount of implicit padding applied on both sides. Default: 0

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

In the simplest case, the output value of the layer with input size
\\(N, C\_{\mbox{in}}, H, W)\\ and output \\(N, C\_{\mbox{out}},
H\_{\mbox{out}}, W\_{\mbox{out}})\\ can be precisely described as:

\$\$ \mbox{out}(N_i, C\_{\mbox{out}\_j}) =
\mbox{bias}(C\_{\mbox{out}\_j}) + \sum\_{k = 0}^{C\_{\mbox{in}} - 1}
\mbox{weight}(C\_{\mbox{out}\_j}, k) \star \mbox{input}(N_i, k) \$\$

where \\\star\\ is the valid 2D cross-correlation operator, \\N\\ is a
batch size, \\C\\ denotes a number of channels, \\H\\ is a height of
input planes in pixels, and \\W\\ is width in pixels.

- `stride` controls the stride for the cross-correlation, a single
  number or a tuple.

- `padding` controls the amount of implicit zero-paddings on both sides
  for `padding` number of points for each dimension.

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
    own set of filters, of size:
    \\\left\lfloor\frac{out\\channels}{in\\channels}\right\rfloor\\.

The parameters `kernel_size`, `stride`, `padding`, `dilation` can either
be:

- a single `int` – in which case the same value is used for the height
  and width dimension

- a `tuple` of two ints – in which case, the first `int` is used for the
  height dimension, and the second `int` for the width dimension

## Note

Depending of the size of your kernel, several (of the last) columns of
the input might be lost, because it is a valid cross-correlation, and
not a full cross-correlation. It is up to the user to add proper
padding.

When `groups == in_channels` and `out_channels == K * in_channels`,
where `K` is a positive integer, this operation is also termed in
literature as depthwise convolution. In other words, for an input of
size :math:`(N, C_{in}, H_{in}, W_{in})`, a depthwise convolution with a
depthwise multiplier `K`, can be constructed by arguments
\\(in\\channels=C\_{in}, out\\channels=C\_{in} \times K, ...,
groups=C\_{in})\\.

In some circumstances when using the CUDA backend with CuDNN, this
operator may select a nondeterministic algorithm to increase
performance. If this is undesirable, you can try to make the operation
deterministic (potentially at a performance cost) by setting
`backends_cudnn_deterministic = TRUE`.

## Shape

- Input: \\(N, C\_{in}, H\_{in}, W\_{in})\\

- Output: \\(N, C\_{out}, H\_{out}, W\_{out})\\ where \$\$ H\_{out} =
  \left\lfloor\frac{H\_{in} + 2 \times \mbox{padding}\[0\] -
  \mbox{dilation}\[0\] \times (\mbox{kernel\\size}\[0\] - 1) -
  1}{\mbox{stride}\[0\]} + 1\right\rfloor \$\$ \$\$ W\_{out} =
  \left\lfloor\frac{W\_{in} + 2 \times \mbox{padding}\[1\] -
  \mbox{dilation}\[1\] \times (\mbox{kernel\\size}\[1\] - 1) -
  1}{\mbox{stride}\[1\]} + 1\right\rfloor \$\$

## Attributes

- weight (Tensor): the learnable weights of the module of shape
  \\(\mbox{out\\channels}, \frac{\mbox{in\\channels}}{\mbox{groups}}\\,
  \\\mbox{kernel\\size\[0\]}, \mbox{kernel\\size\[1\]})\\. The values of
  these weights are sampled from \\\mathcal{U}(-\sqrt{k}, \sqrt{k})\\
  where \\k = \frac{groups}{C\_{\mbox{in}} \*
  \prod\_{i=0}^{1}\mbox{kernel\\size}\[i\]}\\

- bias (Tensor): the learnable bias of the module of shape
  (out_channels). If `bias` is `TRUE`, then the values of these weights
  are sampled from \\\mathcal{U}(-\sqrt{k}, \sqrt{k})\\ where \\k =
  \frac{groups}{C\_{\mbox{in}} \*
  \prod\_{i=0}^{1}\mbox{kernel\\size}\[i\]}\\

## Examples

``` r
if (torch_is_installed()) {

# With square kernels and equal stride
m <- nn_conv2d(16, 33, 3, stride = 2)
# non-square kernels and unequal stride and with padding
m <- nn_conv2d(16, 33, c(3, 5), stride = c(2, 1), padding = c(4, 2))
# non-square kernels and unequal stride and with padding and dilation
m <- nn_conv2d(16, 33, c(3, 5), stride = c(2, 1), padding = c(4, 2), dilation = c(3, 1))
input <- torch_randn(20, 16, 50, 100)
output <- m(input)
}
```
