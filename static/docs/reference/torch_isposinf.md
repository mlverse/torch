# Isposinf

Isposinf

## Usage

``` r
torch_isposinf(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## isposinf(input, \*, out=None) -\> Tensor

Tests if each element of `input` is positive infinity or not.

## Examples

``` r
if (torch_is_installed()) {

a <- torch_tensor(c(-Inf, Inf, 1.2))
torch_isposinf(a)
}
#> torch_tensor
#>  0
#>  1
#>  0
#> [ CPUBoolType{3} ]
```
