# Tanh module

Applies the element-wise function:

## Usage

``` r
nn_tanh()
```

## Details

\$\$ \mbox{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) +
\exp(-x)} \$\$

## Shape

- Input: \\(N, \*)\\ where `*` means, any number of additional
  dimensions

- Output: \\(N, \*)\\, same shape as the input

## Examples

``` r
if (torch_is_installed()) {
m <- nn_tanh()
input <- torch_randn(2)
output <- m(input)
}
```
