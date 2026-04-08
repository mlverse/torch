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
#>  0.2784
#>  0.9747
#> -0.1449
#> -0.2538
#> [ CPUFloatType{4} ]
```
