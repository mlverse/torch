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
#> -1.1332
#> -0.4523
#>  0.7603
#> -0.6719
#> [ CPUFloatType{4} ]
```
