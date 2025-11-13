# Asin

Asin

## Usage

``` r
torch_asin(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## asin(input, out=NULL) -\> Tensor

Returns a new tensor with the arcsine of the elements of `input`.

\$\$ \mbox{out}\_{i} = \sin^{-1}(\mbox{input}\_{i}) \$\$

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_asin(a)
}
#> torch_tensor
#> -0.2418
#>  0.5434
#>  0.4620
#>     nan
#> [ CPUFloatType{4} ]
```
