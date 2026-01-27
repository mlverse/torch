# Log2

Log2

## Usage

``` r
torch_log2(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## log2(input, out=NULL) -\> Tensor

Returns a new tensor with the logarithm to the base 2 of the elements of
`input`.

\$\$ y\_{i} = \log\_{2} (x\_{i}) \$\$

## Examples

``` r
if (torch_is_installed()) {

a = torch_rand(5)
a
torch_log2(a)
}
#> torch_tensor
#> -0.7081
#> -2.5679
#> -0.2283
#> -0.2051
#> -1.2357
#> [ CPUFloatType{5} ]
```
