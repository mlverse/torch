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
#>  1.3895
#>  1.2650
#>  1.2538
#>  2.9975
#> [ CPUFloatType{4} ]
```
