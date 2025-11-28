# Applies a 1D average pooling over an input signal composed of several input planes.

In the simplest case, the output value of the layer with input size
\\(N, C, L)\\, output \\(N, C, L\_{out})\\ and `kernel_size` \\k\\ can
be precisely described as:

## Usage

``` r
nn_avg_pool1d(
  kernel_size,
  stride = NULL,
  padding = 0,
  ceil_mode = FALSE,
  count_include_pad = TRUE
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

## Details

\$\$ \mbox{out}(N_i, C_j, l) = \frac{1}{k} \sum\_{m=0}^{k-1}
\mbox{input}(N_i, C_j, \mbox{stride} \times l + m) \$\$

If `padding` is non-zero, then the input is implicitly zero-padded on
both sides for `padding` number of points.

The parameters `kernel_size`, `stride`, `padding` can each be an `int`
or a one-element tuple.

## Shape

- Input: \\(N, C, L\_{in})\\

- Output: \\(N, C, L\_{out})\\, where

\$\$ L\_{out} = \left\lfloor \frac{L\_{in} + 2 \times \mbox{padding} -
\mbox{kernel\\size}}{\mbox{stride}} + 1\right\rfloor \$\$

## Examples

``` r
if (torch_is_installed()) {

# pool with window of size=3, stride=2
m <- nn_avg_pool1d(3, stride = 2)
m(torch_randn(1, 1, 8))
}
#> torch_tensor
#> (1,.,.) = 
#>   0.1147 -0.1298  0.0522
#> [ CPUFloatType{1,1,3} ]
```
