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
#>  0.5495
#>  2.0529
#>  2.1358
#>  2.3470
#>  2.4288
#>  2.6118
#>  2.6768
#>  3.0053
#>  3.0807
#>  3.1185
#> [ CPUFloatType{10} ]
```
