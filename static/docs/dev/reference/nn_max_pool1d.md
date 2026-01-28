# MaxPool1D module

Applies a 1D max pooling over an input signal composed of several input
planes.

## Usage

``` r
nn_max_pool1d(
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
  [`nn_max_unpool1d()`](https://torch.mlverse.org/docs/dev/reference/nn_max_unpool1d.md)
  later.

- ceil_mode:

  when `TRUE`, will use `ceil` instead of `floor` to compute the output
  shape

## Details

In the simplest case, the output value of the layer with input size
\\(N, C, L)\\ and output \\(N, C, L\_{out})\\ can be precisely described
as:

\$\$ out(N_i, C_j, k) = \max\_{m=0, \ldots, \mbox{kernel\\size} - 1}
input(N_i, C_j, stride \times k + m) \$\$

If `padding` is non-zero, then the input is implicitly zero-padded on
both sides for `padding` number of points. `dilation` controls the
spacing between the kernel points. It is harder to describe, but this
[link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)
has a nice visualization of what `dilation` does.

## Shape

- Input: \\(N, C, L\_{in})\\

- Output: \\(N, C, L\_{out})\\, where

\$\$ L\_{out} = \left\lfloor \frac{L\_{in} + 2 \times \mbox{padding} -
\mbox{dilation} \times (\mbox{kernel\\size} - 1) - 1}{\mbox{stride}} +
1\right\rfloor \$\$

## Examples

``` r
if (torch_is_installed()) {
# pool of size=3, stride=2
m <- nn_max_pool1d(3, stride = 2)
input <- torch_randn(20, 16, 50)
output <- m(input)
}
```
