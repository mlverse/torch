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
#> -0.1598
#>  0.5452
#> -0.1434
#> -0.2577
#> [ CPUFloatType{4} ]
```
