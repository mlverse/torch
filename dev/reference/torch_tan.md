# Tan

Tan

## Usage

``` r
torch_tan(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## tan(input, out=NULL) -\> Tensor

Returns a new tensor with the tangent of the elements of `input`.

\$\$ \mbox{out}\_{i} = \tan(\mbox{input}\_{i}) \$\$

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_tan(a)
}
#> torch_tensor
#> -5.2952
#> -5.0721
#> -3.0901
#>  0.1237
#> [ CPUFloatType{4} ]
```
