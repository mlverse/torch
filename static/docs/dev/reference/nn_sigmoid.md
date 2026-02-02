# Sigmoid module

Applies the element-wise function:

## Usage

``` r
nn_sigmoid()
```

## Details

\$\$ \mbox{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)} \$\$

## Shape

- Input: \\(N, \*)\\ where `*` means, any number of additional
  dimensions

- Output: \\(N, \*)\\, same shape as the input

## Examples

``` r
if (torch_is_installed()) {
m <- nn_sigmoid()
input <- torch_randn(2)
output <- m(input)
}
```
