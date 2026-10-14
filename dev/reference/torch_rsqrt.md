# Rsqrt

Rsqrt

## Usage

``` r
torch_rsqrt(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## rsqrt(input, out=NULL) -\> Tensor

Returns a new tensor with the reciprocal of the square-root of each of
the elements of `input`.

\$\$ \mbox{out}\_{i} = \frac{1}{\sqrt{\mbox{input}\_{i}}} \$\$

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_rsqrt(a)
}
#> torch_tensor
#>  0.9448
#>  4.4903
#>  1.1013
#>  2.4056
#> [ CPUFloatType{4} ]
```
