# Nextafter

Nextafter

## Usage

``` r
torch_nextafter(self, other)
```

## Arguments

- self:

  (Tensor) the first input tensor

- other:

  (Tensor) the second input tensor

## nextafter(input, other, \*, out=None) -\> Tensor

Return the next floating-point value after `input` towards `other`,
elementwise.

The shapes of `input` and `other` must be broadcastable .

## Examples

``` r
if (torch_is_installed()) {

eps <- torch_finfo(torch_float32())$eps
torch_nextafter(torch_tensor(c(1, 2)), torch_tensor(c(2, 1))) == torch_tensor(c(eps + 1, 2 - eps))
}
#> torch_tensor
#>  1
#>  1
#> [ CPUBoolType{2} ]
```
