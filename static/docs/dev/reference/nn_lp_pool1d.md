# Applies a 1D power-average pooling over an input signal composed of several input planes.

On each window, the function computed is:

## Usage

``` r
nn_lp_pool1d(norm_type, kernel_size, stride = NULL, ceil_mode = FALSE)
```

## Arguments

- norm_type:

  if inf than one gets max pooling if 0 you get sum pooling (
  proportional to the avg pooling)

- kernel_size:

  a single int, the size of the window

- stride:

  a single int, the stride of the window. Default value is `kernel_size`

- ceil_mode:

  when TRUE, will use `ceil` instead of `floor` to compute the output
  shape

## Details

\$\$ f(X) = \sqrt\[p\]{\sum\_{x \in X} x^{p}} \$\$

- At p = \\\infty\\, one gets Max Pooling

- At p = 1, one gets Sum Pooling (which is proportional to Average
  Pooling)

## Note

If the sum to the power of `p` is zero, the gradient of this function is
not defined. This implementation will set the gradient to zero in this
case.

## Shape

- Input: \\(N, C, L\_{in})\\

- Output: \\(N, C, L\_{out})\\, where

\$\$ L\_{out} = \left\lfloor\frac{L\_{in} -
\mbox{kernel\\size}}{\mbox{stride}} + 1\right\rfloor \$\$

## Examples

``` r
if (torch_is_installed()) {
# power-2 pool of window of length 3, with stride 2.
m <- nn_lp_pool1d(2, 3, stride = 2)
input <- torch_randn(20, 16, 50)
output <- m(input)
}
```
