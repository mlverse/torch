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
#>  1.5933
#>  2.2874
#>     nan
#>  1.7637
#> [ CPUFloatType{4} ]
```
