# Threshold module

Thresholds each element of the input Tensor.

## Usage

``` r
nn_threshold(threshold, value, inplace = FALSE)
```

## Arguments

- threshold:

  The value to threshold at

- value:

  The value to replace with

- inplace:

  can optionally do the operation in-place. Default: `FALSE`

## Details

Threshold is defined as: \$\$ y = \left\\ \begin{array}{ll} x, &\mbox{
if } x \> \mbox{threshold} \\ \mbox{value}, &\mbox{ otherwise }
\end{array} \right. \$\$

## Shape

- Input: \\(N, \*)\\ where `*` means, any number of additional
  dimensions

- Output: \\(N, \*)\\, same shape as the input

## Examples

``` r
if (torch_is_installed()) {
m <- nn_threshold(0.1, 20)
input <- torch_randn(2)
output <- m(input)
}
```
