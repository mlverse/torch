# LogSoftmax module

Applies the \\\log(\mbox{Softmax}(x))\\ function to an n-dimensional
input Tensor. The LogSoftmax formulation can be simplified as:

## Usage

``` r
nn_log_softmax(dim)
```

## Arguments

- dim:

  (int): A dimension along which LogSoftmax will be computed.

## Value

a Tensor of the same dimension and shape as the input with values in the
range \[-inf, 0)

## Details

\$\$ \mbox{LogSoftmax}(x\_{i}) = \log\left(\frac{\exp(x_i) }{ \sum_j
\exp(x_j)} \right) \$\$

## Shape

- Input: \\(\*)\\ where `*` means, any number of additional dimensions

- Output: \\(\*)\\, same shape as the input

## Examples

``` r
if (torch_is_installed()) {
m <- nn_log_softmax(1)
input <- torch_randn(2, 3)
output <- m(input)
}
```
