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
#> -0.9161
#>  0.5418
#> -0.4674
#>  0.5808
#> [ CPUFloatType{4} ]
```
