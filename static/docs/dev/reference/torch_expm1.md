# Expm1

Expm1

## Usage

``` r
torch_expm1(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## expm1(input, out=NULL) -\> Tensor

Returns a new tensor with the exponential of the elements minus 1 of
`input`.

\$\$ y\_{i} = e^{x\_{i}} - 1 \$\$

## Examples

``` r
if (torch_is_installed()) {

torch_expm1(torch_tensor(c(0, log(2))))
}
#> torch_tensor
#>  0
#>  1
#> [ CPUFloatType{2} ]
```
