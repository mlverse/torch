# Asinh

Asinh

## Usage

``` r
torch_asinh(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## asinh(input, \*, out=None) -\> Tensor

Returns a new tensor with the inverse hyperbolic sine of the elements of
`input`.

\$\$ \mbox{out}\_{i} = \sinh^{-1}(\mbox{input}\_{i}) \$\$

## Examples

``` r
if (torch_is_installed()) {

a <- torch_randn(c(4))
a
torch_asinh(a)
}
#> torch_tensor
#> -0.8294
#>  1.0604
#>  0.4480
#>  0.4399
#> [ CPUFloatType{4} ]
```
