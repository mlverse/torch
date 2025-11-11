# ReLu6 module

Applies the element-wise function:

## Usage

``` r
nn_relu6(inplace = FALSE)
```

## Arguments

- inplace:

  can optionally do the operation in-place. Default: `FALSE`

## Details

\$\$ \mbox{ReLU6}(x) = \min(\max(0,x), 6) \$\$

## Shape

- Input: \\(N, \*)\\ where `*` means, any number of additional
  dimensions

- Output: \\(N, \*)\\, same shape as the input

## Examples

``` r
if (torch_is_installed()) {
m <- nn_relu6()
input <- torch_randn(2)
output <- m(input)
}
```
