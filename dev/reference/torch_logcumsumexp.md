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
#>  0.5850
#>  1.0176
#>  1.2228
#>  1.4115
#>  3.4432
#>  3.6573
#>  3.7099
#>  3.7310
#>  3.7376
#>  3.7507
#> [ CPUFloatType{10} ]
```
