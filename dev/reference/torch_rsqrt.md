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
#>  2.3835
#>     nan
#>  4.1657
#>  1.1193
#> [ CPUFloatType{4} ]
```
