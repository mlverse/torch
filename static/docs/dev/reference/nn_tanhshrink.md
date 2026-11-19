# Tanhshrink module

Applies the element-wise function:

## Usage

``` r
nn_tanhshrink()
```

## Details

\$\$ \mbox{Tanhshrink}(x) = x - \tanh(x) \$\$

## Shape

- Input: \\(N, \*)\\ where `*` means, any number of additional
  dimensions

- Output: \\(N, \*)\\, same shape as the input

## Examples

``` r
if (torch_is_installed()) {
m <- nn_tanhshrink()
input <- torch_randn(2)
output <- m(input)
}
```
