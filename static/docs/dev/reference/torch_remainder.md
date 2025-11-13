# Remainder

Remainder

## Usage

``` r
torch_remainder(self, other)
```

## Arguments

- self:

  (Tensor) the dividend

- other:

  (Tensor or float) the divisor that may be either a number or a Tensor
  of the same shape as the dividend

## remainder(input, other, out=NULL) -\> Tensor

Computes the element-wise remainder of division.

The divisor and dividend may contain both for integer and floating point
numbers. The remainder has the same sign as the divisor.

When `other` is a tensor, the shapes of `input` and `other` must be
broadcastable .

## Examples

``` r
if (torch_is_installed()) {

torch_remainder(torch_tensor(c(-3., -2, -1, 1, 2, 3)), 2)
torch_remainder(torch_tensor(c(1., 2, 3, 4, 5)), 1.5)
}
#> torch_tensor
#>  1.0000
#>  0.5000
#>  0.0000
#>  1.0000
#>  0.5000
#> [ CPUFloatType{5} ]
```
