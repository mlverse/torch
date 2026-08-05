# Softmax module

Applies the Softmax function to an n-dimensional input Tensor rescaling
them so that the elements of the n-dimensional output Tensor lie in the
range `[0,1]` and sum to 1. Softmax is defined as:

## Usage

``` r
nn_softmax(dim)
```

## Arguments

- dim:

  (int): A dimension along which Softmax will be computed (so every
  slice along dim will sum to 1).

## Value

: a Tensor of the same dimension and shape as the input with values in
the range `[0, 1]`

## Details

\$\$ \mbox{Softmax}(x\_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)} \$\$

When the input Tensor is a sparse tensor then the unspecifed values are
treated as `-Inf`.

## Note

This module doesn't work directly with NLLLoss, which expects the Log to
be computed between the Softmax and itself. Use `LogSoftmax` instead
(it's faster and has better numerical properties).

## Shape

- Input: \\(\*)\\ where `*` means, any number of additional dimensions

- Output: \\(\*)\\, same shape as the input

## Examples

``` r
if (torch_is_installed()) {
m <- nn_softmax(1)
input <- torch_randn(2, 3)
output <- m(input)
}
```
