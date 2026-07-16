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
#> -0.9680
#> -0.4187
#> -0.1491
#>  0.9079
#> -2.4589
#> [ CPUFloatType{5} ]
```
