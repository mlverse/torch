# Applies a 2D power-average pooling over an input signal composed of several input planes.

On each window, the function computed is:

## Usage

``` r
nn_lp_pool2d(norm_type, kernel_size, stride = NULL, ceil_mode = FALSE)
```

## Arguments

- norm_type:

  if inf than one gets max pooling if 0 you get sum pooling (
  proportional to the avg pooling)

- kernel_size:

  the size of the window

- stride:

  the stride of the window. Default value is `kernel_size`

- ceil_mode:

  when TRUE, will use `ceil` instead of `floor` to compute the output
  shape

## Details

\$\$ f(X) = \sqrt\[p\]{\sum\_{x \in X} x^{p}} \$\$

- At p = \\\infty\\, one gets Max Pooling

- At p = 1, one gets Sum Pooling (which is proportional to average
  pooling)

The parameters `kernel_size`, `stride` can either be:

- a single `int` – in which case the same value is used for the height
  and width dimension

- a `tuple` of two ints – in which case, the first `int` is used for the
  height dimension, and the second `int` for the width dimension

## Note

If the sum to the power of `p` is zero, the gradient of this function is
not defined. This implementation will set the gradient to zero in this
case.

## Shape

- Input: \\(N, C, H\_{in}, W\_{in})\\

- Output: \\(N, C, H\_{out}, W\_{out})\\, where

\$\$ H\_{out} = \left\lfloor\frac{H\_{in} -
\mbox{kernel\\size}\[0\]}{\mbox{stride}\[0\]} + 1\right\rfloor \$\$ \$\$
W\_{out} = \left\lfloor\frac{W\_{in} -
\mbox{kernel\\size}\[1\]}{\mbox{stride}\[1\]} + 1\right\rfloor \$\$

## Examples

``` r
if (torch_is_installed()) {

# power-2 pool of square window of size=3, stride=2
m <- nn_lp_pool2d(2, 3, stride = 2)
# pool of non-square window of power 1.2
m <- nn_lp_pool2d(1.2, c(3, 2), stride = c(2, 1))
input <- torch_randn(20, 16, 50, 32)
output <- m(input)
}
```
