# Bincount

Bincount

## Arguments

- self:

  (Tensor) 1-d int tensor

- weights:

  (Tensor) optional, weight for each value in the input tensor. Should
  be of same size as input tensor.

- minlength:

  (int) optional, minimum number of bins. Should be non-negative.

## bincount(input, weights=NULL, minlength=0) -\> Tensor

Count the frequency of each value in an array of non-negative ints.

The number of bins (size 1) is one larger than the largest value in
`input` unless `input` is empty, in which case the result is a tensor of
size 0. If `minlength` is specified, the number of bins is at least
`minlength` and if `input` is empty, then the result is tensor of size
`minlength` filled with zeros. If `n` is the value at position `i`,
`out[n] += weights[i]` if `weights` is specified else `out[n] += 1`.

.. include:: cuda_deterministic.rst

## Examples

``` r
if (torch_is_installed()) {

input = torch_randint(1, 8, list(5), dtype=torch_int64())
weights = torch_linspace(0, 1, steps=5)
input
weights
torch_bincount(input, weights)
input$bincount(weights)
}
#> torch_tensor
#>  1.7500
#>  0.0000
#>  0.0000
#>  0.5000
#>  0.0000
#>  0.2500
#>  0.0000
#> [ CPUFloatType{7} ]
```
