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
#> -1.1556
#>  0.1619
#>  0.5995
#>  0.9465
#>  1.0681
#>  1.2765
#>  1.5118
#>  2.2099
#>  2.2875
#>  2.4268
#> [ CPUFloatType{10} ]
```
