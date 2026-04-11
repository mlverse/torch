# Fmod

Fmod

## Usage

``` r
torch_fmod(self, other)
```

## Arguments

- self:

  (Tensor) the dividend

- other:

  (Tensor or float) the divisor, which may be either a number or a
  tensor of the same shape as the dividend

## fmod(input, other, out=NULL) -\> Tensor

Computes the element-wise remainder of division.

The dividend and divisor may contain both for integer and floating point
numbers. The remainder has the same sign as the dividend `input`.

When `other` is a tensor, the shapes of `input` and `other` must be
broadcastable .

## Examples

``` r
if (torch_is_installed()) {

torch_fmod(torch_tensor(c(-3., -2, -1, 1, 2, 3)), 2)
torch_fmod(torch_tensor(c(1., 2, 3, 4, 5)), 1.5)
}
#> torch_tensor
#>  1.0000
#>  0.5000
#>  0.0000
#>  1.0000
#>  0.5000
#> [ CPUFloatType{5} ]
```
