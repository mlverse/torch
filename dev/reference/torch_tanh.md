# Tanh

Tanh

## Usage

``` r
torch_tanh(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## tanh(input, out=NULL) -\> Tensor

Returns a new tensor with the hyperbolic tangent of the elements of
`input`.

\$\$ \mbox{out}\_{i} = \tanh(\mbox{input}\_{i}) \$\$

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_tanh(a)
}
#> torch_tensor
#>  0.2215
#> -0.5265
#> -0.3723
#> -0.4054
#> [ CPUFloatType{4} ]
```
