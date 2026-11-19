# MaxPool2D module

Applies a 2D max pooling over an input signal composed of several input
planes.

## Usage

``` r
nn_max_pool2d(
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

  implicit zero padding to be added on both sides

- dilation:

  a parameter that controls the stride of elements in the window

- return_indices:

  if `TRUE`, will return the max indices along with the outputs. Useful
  for
  [`nn_max_unpool2d()`](https://torch.mlverse.org/docs/dev/reference/nn_max_unpool2d.md)
  later.

- ceil_mode:

  when `TRUE`, will use `ceil` instead of `floor` to compute the output
  shape

## Details

In the simplest case, the output value of the layer with input size
\\(N, C, H, W)\\, output \\(N, C, H\_{out}, W\_{out})\\ and
`kernel_size` \\(kH, kW)\\ can be precisely described as:

\$\$ \begin{array}{ll} out(N_i, C_j, h, w) ={} & \max\_{m=0, \ldots,
kH-1} \max\_{n=0, \ldots, kW-1} \\ & \mbox{input}(N_i, C_j,
\mbox{stride\[0\]} \times h + m, \mbox{stride\[1\]} \times w + n)
\end{array} \$\$

If `padding` is non-zero, then the input is implicitly zero-padded on
both sides for `padding` number of points. `dilation` controls the
spacing between the kernel points. It is harder to describe, but this
`link` has a nice visualization of what `dilation` does.

The parameters `kernel_size`, `stride`, `padding`, `dilation` can either
be:

- a single `int` – in which case the same value is used for the height
  and width dimension

- a `tuple` of two ints – in which case, the first `int` is used for the
  height dimension, and the second `int` for the width dimension

## Shape

- Input: \\(N, C, H\_{in}, W\_{in})\\

- Output: \\(N, C, H\_{out}, W\_{out})\\, where

\$\$ H\_{out} = \left\lfloor\frac{H\_{in} + 2 \* \mbox{padding\[0\]} -
\mbox{dilation\[0\]} \times (\mbox{kernel\\size\[0\]} - 1) -
1}{\mbox{stride\[0\]}} + 1\right\rfloor \$\$

\$\$ W\_{out} = \left\lfloor\frac{W\_{in} + 2 \* \mbox{padding\[1\]} -
\mbox{dilation\[1\]} \times (\mbox{kernel\\size\[1\]} - 1) -
1}{\mbox{stride\[1\]}} + 1\right\rfloor \$\$

## Examples

``` r
if (torch_is_installed()) {
# pool of square window of size=3, stride=2
m <- nn_max_pool2d(3, stride = 2)
# pool of non-square window
m <- nn_max_pool2d(c(3, 2), stride = c(2, 1))
input <- torch_randn(20, 16, 50, 32)
output <- m(input)
}
```
