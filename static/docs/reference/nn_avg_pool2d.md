# Applies a 2D average pooling over an input signal composed of several input planes.

In the simplest case, the output value of the layer with input size
\\(N, C, H, W)\\, output \\(N, C, H\_{out}, W\_{out})\\ and
`kernel_size` \\(kH, kW)\\ can be precisely described as:

## Usage

``` r
nn_avg_pool2d(
  kernel_size,
  stride = NULL,
  padding = 0,
  ceil_mode = FALSE,
  count_include_pad = TRUE,
  divisor_override = NULL
)
```

## Arguments

- kernel_size:

  the size of the window

- stride:

  the stride of the window. Default value is `kernel_size`

- padding:

  implicit zero padding to be added on both sides

- ceil_mode:

  when TRUE, will use `ceil` instead of `floor` to compute the output
  shape

- count_include_pad:

  when TRUE, will include the zero-padding in the averaging calculation

- divisor_override:

  if specified, it will be used as divisor, otherwise `kernel_size` will
  be used

## Details

\$\$ out(N_i, C_j, h, w) = \frac{1}{kH \* kW} \sum\_{m=0}^{kH-1}
\sum\_{n=0}^{kW-1} input(N_i, C_j, stride\[0\] \times h + m, stride\[1\]
\times w + n) \$\$

If `padding` is non-zero, then the input is implicitly zero-padded on
both sides for `padding` number of points.

The parameters `kernel_size`, `stride`, `padding` can either be:

- a single `int` – in which case the same value is used for the height
  and width dimension

- a `tuple` of two ints – in which case, the first `int` is used for the
  height dimension, and the second `int` for the width dimension

## Shape

- Input: \\(N, C, H\_{in}, W\_{in})\\

- Output: \\(N, C, H\_{out}, W\_{out})\\, where

\$\$ H\_{out} = \left\lfloor\frac{H\_{in} + 2 \times
\mbox{padding}\[0\] - \mbox{kernel\\size}\[0\]}{\mbox{stride}\[0\]} +
1\right\rfloor \$\$ \$\$ W\_{out} = \left\lfloor\frac{W\_{in} + 2 \times
\mbox{padding}\[1\] - \mbox{kernel\\size}\[1\]}{\mbox{stride}\[1\]} +
1\right\rfloor \$\$

## Examples

``` r
if (torch_is_installed()) {

# pool of square window of size=3, stride=2
m <- nn_avg_pool2d(3, stride = 2)
# pool of non-square window
m <- nn_avg_pool2d(c(3, 2), stride = c(2, 1))
input <- torch_randn(20, 16, 50, 32)
output <- m(input)
}
```
