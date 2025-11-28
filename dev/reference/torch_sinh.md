# Sinh

Sinh

## Usage

``` r
torch_sinh(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## sinh(input, out=NULL) -\> Tensor

Returns a new tensor with the hyperbolic sine of the elements of
`input`.

\$\$ \mbox{out}\_{i} = \sinh(\mbox{input}\_{i}) \$\$

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_sinh(a)
}
#> torch_tensor
#>  0.9243
#> -0.2272
#> -2.3120
#> -0.0070
#> [ CPUFloatType{4} ]
```
