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
#> -12.7620
#>   5.9304
#>   0.2686
#>  -5.5748
#> [ CPUFloatType{4} ]
```
