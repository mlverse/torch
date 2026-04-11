# Softmin

Applies the Softmin function to an n-dimensional input Tensor rescaling
them so that the elements of the n-dimensional output Tensor lie in the
range `[0, 1]` and sum to 1. Softmin is defined as:

## Usage

``` r
nn_softmin(dim)
```

## Arguments

- dim:

  (int): A dimension along which Softmin will be computed (so every
  slice along dim will sum to 1).

## Value

a Tensor of the same dimension and shape as the input, with values in
the range `[0, 1]`.

## Details

\$\$ \mbox{Softmin}(x\_{i}) = \frac{\exp(-x_i)}{\sum_j \exp(-x_j)} \$\$

## Shape

- Input: \\(\*)\\ where `*` means, any number of additional dimensions

- Output: \\(\*)\\, same shape as the input

## Examples

``` r
if (torch_is_installed()) {
m <- nn_softmin(dim = 1)
input <- torch_randn(2, 2)
output <- m(input)
}
```
