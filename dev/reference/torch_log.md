# Log

Log

## Usage

``` r
torch_log(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## log(input, out=NULL) -\> Tensor

Returns a new tensor with the natural logarithm of the elements of
`input`.

\$\$ y\_{i} = \log\_{e} (x\_{i}) \$\$

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(5))
a
torch_log(a)
}
#> torch_tensor
#>     nan
#>     nan
#>  1.0564
#>     nan
#> -0.9264
#> [ CPUFloatType{5} ]
```
