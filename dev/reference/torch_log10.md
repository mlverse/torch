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
#> 0.01 *
#> -6.7682
#> -96.9215
#> -79.9206
#> -58.7613
#> -11.8632
#> [ CPUFloatType{5} ]
```
