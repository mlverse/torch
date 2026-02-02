# Gcd

Gcd

## Usage

``` r
torch_gcd(self, other)
```

## Arguments

- self:

  (Tensor) the input tensor.

- other:

  (Tensor) the second input tensor

## Note

This defines \\gcd(0, 0) = 0\\.

## gcd(input, other, \*, out=None) -\> Tensor

Computes the element-wise greatest common divisor (GCD) of `input` and
`other`.

Both `input` and `other` must have integer types.

## Examples

``` r
if (torch_is_installed()) {

if (torch::cuda_is_available()) {
a <- torch_tensor(c(5, 10, 15), dtype = torch_long(), device = "cuda")
b <- torch_tensor(c(3, 4, 5), dtype = torch_long(), device = "cuda")
torch_gcd(a, b)
c <- torch_tensor(c(3L), device = "cuda")
torch_gcd(a, c)
}
}
```
