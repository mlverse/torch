# Isneginf

Isneginf

## Usage

``` r
torch_isneginf(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## isneginf(input, \*, out=None) -\> Tensor

Tests if each element of `input` is negative infinity or not.

## Examples

``` r
if (torch_is_installed()) {

a <- torch_tensor(c(-Inf, Inf, 1.2))
torch_isneginf(a)
}
#> torch_tensor
#>  1
#>  0
#>  0
#> [ CPUBoolType{3} ]
```
