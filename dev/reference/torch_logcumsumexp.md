# Logcumsumexp

Logcumsumexp

## Usage

``` r
torch_logcumsumexp(self, dim)
```

## Arguments

- self:

  (Tensor) the input tensor.

- dim:

  (int) the dimension to do the operation over

## logcumsumexp(input, dim, \*, out=None) -\> Tensor

Returns the logarithm of the cumulative summation of the exponentiation
of elements of `input` in the dimension `dim`.

For summation index \\j\\ given by `dim` and other indices \\i\\, the
result is

\$\$ \mbox{logcumsumexp}(x)\_{ij} = \log \sum\limits\_{j=0}^{i}
\exp(x\_{ij}) \$\$

## Examples

``` r
if (torch_is_installed()) {

a <- torch_randn(c(10))
torch_logcumsumexp(a, dim=1)
}
#> torch_tensor
#> -0.7306
#>  1.3275
#>  1.3342
#>  1.5334
#>  1.6963
#>  2.8717
#>  2.9340
#>  2.9722
#>  2.9890
#>  3.0233
#> [ CPUFloatType{10} ]
```
