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
#>  1.6436
#>     nan
#>  1.7434
#>  1.4091
#> [ CPUFloatType{4} ]
```
