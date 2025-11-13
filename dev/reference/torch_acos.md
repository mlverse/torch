# Acos

Acos

## Usage

``` r
torch_acos(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## acos(input) -\> Tensor

Returns a new tensor with the arccosine of the elements of `input`.

\$\$ \mbox{out}\_{i} = \cos^{-1}(\mbox{input}\_{i}) \$\$

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_acos(a)
}
#> torch_tensor
#>  1.6275
#>  0.8731
#>  2.0021
#>  1.2803
#> [ CPUFloatType{4} ]
```
