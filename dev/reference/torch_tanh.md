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
#> -0.4770
#>  0.1966
#>  0.9671
#>  0.6407
#> [ CPUFloatType{4} ]
```
