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
#> -1.1546
#>  0.7911
#> -0.7281
#>  1.2066
#> [ CPUFloatType{4} ]
```
