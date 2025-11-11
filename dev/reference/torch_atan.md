# Atan

Atan

## Usage

``` r
torch_atan(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## atan(input, out=NULL) -\> Tensor

Returns a new tensor with the arctangent of the elements of `input`.

\$\$ \mbox{out}\_{i} = \tan^{-1}(\mbox{input}\_{i}) \$\$

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_atan(a)
}
#> torch_tensor
#>  0.7122
#>  0.8488
#> -0.3526
#> -0.9515
#> [ CPUFloatType{4} ]
```
