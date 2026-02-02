# Applies a 3D max pooling over an input signal composed of several input planes.

In the simplest case, the output value of the layer with input size
\\(N, C, D, H, W)\\, output \\(N, C, D\_{out}, H\_{out}, W\_{out})\\ and
`kernel_size` \\(kD, kH, kW)\\ can be precisely described as:

## Usage

``` r
nn_max_pool3d(
  kernel_size,
  stride = NULL,
  padding = 0,
  dilation = 1,
  return_indices = FALSE,
  ceil_mode = FALSE
)
```

## Arguments

- kernel_size:

  the size of the window to take a max over

- stride:

  the stride of the window. Default value is `kernel_size`

- padding:

  implicit zero padding to be added on all three sides

- dilation:

  a parameter that controls the stride of elements in the window

- return_indices:

  if `TRUE`, will return the max indices along with the outputs. Useful
  for `torch_nn.MaxUnpool3d` later

- ceil_mode:

  when TRUE, will use `ceil` instead of `floor` to compute the output
  shape

## Details

\$\$ \begin{array}{ll} \mbox{out}(N_i, C_j, d, h, w) = & \max\_{k=0,
\ldots, kD-1} \max\_{m=0, \ldots, kH-1} \max\_{n=0, \ldots, kW-1} \\ &
\mbox{input}(N_i, C_j, \mbox{stride\[0\]} \times d + k,
\mbox{stride\[1\]} \times h + m, \mbox{stride\[2\]} \times w + n)
\end{array} \$\$

If `padding` is non-zero, then the input is implicitly zero-padded on
both sides for `padding` number of points. `dilation` controls the
spacing between the kernel points. It is harder to describe, but this
`link`\_ has a nice visualization of what `dilation` does. The
parameters `kernel_size`, `stride`, `padding`, `dilation` can either be:

- a single `int` – in which case the same value is used for the depth,
  height and width dimension

- a `tuple` of three ints – in which case, the first `int` is used for
  the depth dimension, the second `int` for the height dimension and the
  third `int` for the width dimension

## Shape

- Input: \\(N, C, D\_{in}, H\_{in}, W\_{in})\\

- Output: \\(N, C, D\_{out}, H\_{out}, W\_{out})\\, where \$\$ D\_{out}
  = \left\lfloor\frac{D\_{in} + 2 \times \mbox{padding}\[0\] -
  \mbox{dilation}\[0\] \times (\mbox{kernel\\size}\[0\] - 1) -
  1}{\mbox{stride}\[0\]} + 1\right\rfloor \$\$

\$\$ H\_{out} = \left\lfloor\frac{H\_{in} + 2 \times
\mbox{padding}\[1\] - \mbox{dilation}\[1\] \times
(\mbox{kernel\\size}\[1\] - 1) - 1}{\mbox{stride}\[1\]} + 1\right\rfloor
\$\$

\$\$ W\_{out} = \left\lfloor\frac{W\_{in} + 2 \times
\mbox{padding}\[2\] - \mbox{dilation}\[2\] \times
(\mbox{kernel\\size}\[2\] - 1) - 1}{\mbox{stride}\[2\]} + 1\right\rfloor
\$\$

## Examples

``` r
if (torch_is_installed()) {
# pool of square window of size=3, stride=2
m <- nn_max_pool3d(3, stride = 2)
# pool of non-square window
m <- nn_max_pool3d(c(3, 2, 2), stride = c(2, 1, 2))
input <- torch_randn(20, 16, 50, 44, 31)
output <- m(input)
}
```
