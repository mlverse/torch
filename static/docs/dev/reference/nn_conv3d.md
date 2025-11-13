# Conv3D module

Applies a 3D convolution over an input signal composed of several input
planes. In the simplest case, the output value of the layer with input
size \\(N, C\_{in}, D, H, W)\\ and output \\(N, C\_{out}, D\_{out},
H\_{out}, W\_{out})\\ can be precisely described as:

## Usage

``` r
nn_conv3d(
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

  (int, tuple or str, optional): padding added to all six sides of the
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

\$\$ out(N_i, C\_{out_j}) = bias(C\_{out_j}) + \sum\_{k = 0}^{C\_{in} -
1} weight(C\_{out_j}, k) \star input(N_i, k) \$\$

where \\\star\\ is the valid 3D `cross-correlation` operator

- `stride` controls the stride for the cross-correlation.

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

- At groups= `in_channels`, each input channel is convolved with its own
  set of filters, of size
  \\\left\lfloor\frac{out\\channels}{in\\channels}\right\rfloor\\.

The parameters `kernel_size`, `stride`, `padding`, `dilation` can either
be:

- a single `int` – in which case the same value is used for the depth,
  height and width dimension

- a `tuple` of three ints – in which case, the first `int` is used for
  the depth dimension, the second `int` for the height dimension and the
  third `int` for the width dimension

## Note

Depending of the size of your kernel, several (of the last) columns of
the input might be lost, because it is a valid `cross-correlation`*, and
not a full `cross-correlation`*. It is up to the user to add proper
padding.

When `groups == in_channels` and `out_channels == K * in_channels`,
where `K` is a positive integer, this operation is also termed in
literature as depthwise convolution. In other words, for an input of
size \\(N, C\_{in}, D\_{in}, H\_{in}, W\_{in})\\, a depthwise
convolution with a depthwise multiplier `K`, can be constructed by
arguments \\(in\\channels=C\_{in}, out\\channels=C\_{in} \times K, ...,
groups=C\_{in})\\.

In some circumstances when using the CUDA backend with CuDNN, this
operator may select a nondeterministic algorithm to increase
performance. If this is undesirable, you can try to make the operation
deterministic (potentially at a performance cost) by setting
`torch.backends.cudnn.deterministic = TRUE`. Please see the notes on
:doc:`/notes/randomness` for background.

## Shape

- Input: \\(N, C\_{in}, D\_{in}, H\_{in}, W\_{in})\\

- Output: \\(N, C\_{out}, D\_{out}, H\_{out}, W\_{out})\\ where \$\$
  D\_{out} = \left\lfloor\frac{D\_{in} + 2 \times \mbox{padding}\[0\] -
  \mbox{dilation}\[0\] \times (\mbox{kernel\\size}\[0\] - 1) -
  1}{\mbox{stride}\[0\]} + 1\right\rfloor \$\$ \$\$ H\_{out} =
  \left\lfloor\frac{H\_{in} + 2 \times \mbox{padding}\[1\] -
  \mbox{dilation}\[1\] \times (\mbox{kernel\\size}\[1\] - 1) -
  1}{\mbox{stride}\[1\]} + 1\right\rfloor \$\$ \$\$ W\_{out} =
  \left\lfloor\frac{W\_{in} + 2 \times \mbox{padding}\[2\] -
  \mbox{dilation}\[2\] \times (\mbox{kernel\\size}\[2\] - 1) -
  1}{\mbox{stride}\[2\]} + 1\right\rfloor \$\$

## Attributes

- weight (Tensor): the learnable weights of the module of shape
  \\(\mbox{out\\channels}, \frac{\mbox{in\\channels}}{\mbox{groups}},\\
  \\\mbox{kernel\\size\[0\]}, \mbox{kernel\\size\[1\]},
  \mbox{kernel\\size\[2\]})\\. The values of these weights are sampled
  from \\\mathcal{U}(-\sqrt{k}, \sqrt{k})\\ where \\k =
  \frac{groups}{C\_{\mbox{in}} \*
  \prod\_{i=0}^{2}\mbox{kernel\\size}\[i\]}\\

- bias (Tensor): the learnable bias of the module of shape
  (out_channels). If `bias` is `True`, then the values of these weights
  are sampled from \\\mathcal{U}(-\sqrt{k}, \sqrt{k})\\ where \\k =
  \frac{groups}{C\_{\mbox{in}} \*
  \prod\_{i=0}^{2}\mbox{kernel\\size}\[i\]}\\

## Examples

``` r
if (torch_is_installed()) {
# With square kernels and equal stride
m <- nn_conv3d(16, 33, 3, stride = 2)
# non-square kernels and unequal stride and with padding
m <- nn_conv3d(16, 33, c(3, 5, 2), stride = c(2, 1, 1), padding = c(4, 2, 0))
input <- torch_randn(20, 16, 10, 50, 100)
output <- m(input)
}
```
