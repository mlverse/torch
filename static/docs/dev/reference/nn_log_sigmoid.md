# LogSigmoid module

Applies the element-wise function: \$\$ \mbox{LogSigmoid}(x) =
\log\left(\frac{ 1 }{ 1 + \exp(-x)}\right) \$\$

## Usage

``` r
nn_log_sigmoid()
```

## Shape

- Input: \\(N, \*)\\ where `*` means, any number of additional
  dimensions

- Output: \\(N, \*)\\, same shape as the input

## Examples

``` r
if (torch_is_installed()) {
m <- nn_log_sigmoid()
input <- torch_randn(2)
output <- m(input)
}
```
