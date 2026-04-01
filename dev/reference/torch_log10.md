# Log10

Log10

## Usage

``` r
torch_log10(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## log10(input, out=NULL) -\> Tensor

Returns a new tensor with the logarithm to the base 10 of the elements
of `input`.

\$\$ y\_{i} = \log\_{10} (x\_{i}) \$\$

## Examples

``` r
if (torch_is_installed()) {

a = torch_rand(5)
a
torch_log10(a)
}
#> torch_tensor
#> -0.4921
#> -0.4191
#> -0.2006
#> -0.4563
#> -1.0204
#> [ CPUFloatType{5} ]
```
