# Cosh

Cosh

## Usage

``` r
torch_cosh(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## cosh(input, out=NULL) -\> Tensor

Returns a new tensor with the hyperbolic cosine of the elements of
`input`.

\$\$ \mbox{out}\_{i} = \cosh(\mbox{input}\_{i}) \$\$

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_cosh(a)
}
#> torch_tensor
#>  2.4432
#>  2.8574
#>  1.7522
#>  1.0822
#> [ CPUFloatType{4} ]
```
