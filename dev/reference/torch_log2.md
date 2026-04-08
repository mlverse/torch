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
#> -1.7350
#> -3.2301
#> -0.9482
#> -0.1054
#> -1.2574
#> [ CPUFloatType{5} ]
```
