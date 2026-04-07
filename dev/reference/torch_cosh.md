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
#>  2.3245
#>  1.0843
#>  1.5372
#>  1.8875
#> [ CPUFloatType{4} ]
```
