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
#>  1.0274
#> -0.3623
#>     nan
#> -0.2678
#> [ CPUFloatType{5} ]
```
