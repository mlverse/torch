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
#> -0.2491
#>  0.3657
#> -0.1447
#>  0.9221
#> [ CPUFloatType{4} ]
```
