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
#>  1.3169
#> -1.1311
#>  1.6403
#> -0.1735
#> [ CPUFloatType{4} ]
```
