# Logaddexp

Logaddexp

## Usage

``` r
torch_logaddexp(self, other)
```

## Arguments

- self:

  (Tensor) the input tensor.

- other:

  (Tensor) the second input tensor

## logaddexp(input, other, \*, out=None) -\> Tensor

Logarithm of the sum of exponentiations of the inputs.

Calculates pointwise \\\log\left(e^x + e^y\right)\\. This function is
useful in statistics where the calculated probabilities of events may be
so small as to exceed the range of normal floating point numbers. In
such cases the logarithm of the calculated probability is stored. This
function allows adding probabilities stored in such a fashion.

This op should be disambiguated with
[`torch_logsumexp()`](https://torch.mlverse.org/docs/dev/reference/torch_logsumexp.md)
which performs a reduction on a single tensor.

## Examples

``` r
if (torch_is_installed()) {

torch_logaddexp(torch_tensor(c(-1.0)), torch_tensor(c(-1.0, -2, -3)))
torch_logaddexp(torch_tensor(c(-100.0, -200, -300)), torch_tensor(c(-1.0, -2, -3)))
torch_logaddexp(torch_tensor(c(1.0, 2000, 30000)), torch_tensor(c(-1.0, -2, -3)))
}
#> torch_tensor
#>      1.1269
#>   2000.0000
#>  30000.0000
#> [ CPUFloatType{3} ]
```
