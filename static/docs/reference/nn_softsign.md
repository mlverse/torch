# Softsign module

Applies the element-wise function: \$\$ \mbox{SoftSign}(x) = \frac{x}{
1 + \|x\|} \$\$

## Usage

``` r
nn_softsign()
```

## Shape

- Input: \\(N, \*)\\ where `*` means, any number of additional
  dimensions

- Output: \\(N, \*)\\, same shape as the input

## Examples

``` r
if (torch_is_installed()) {
m <- nn_softsign()
input <- torch_randn(2)
output <- m(input)
}
```
