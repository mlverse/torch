# Exp2

Exp2

## Usage

``` r
torch_exp2(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## exp2(input, \*, out=None) -\> Tensor

Computes the base two exponential function of `input`.

\$\$ y\_{i} = 2^{x\_{i}} \$\$

## Examples

``` r
if (torch_is_installed()) {

torch_exp2(torch_tensor(c(0, log2(2.), 3, 4)))
}
#> torch_tensor
#>   1
#>   2
#>   8
#>  16
#> [ CPUFloatType{4} ]
```
