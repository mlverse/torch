# Signbit

Signbit

## Usage

``` r
torch_signbit(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## signbit(input, \*, out=None) -\> Tensor

Tests if each element of `input` has its sign bit set (is less than
zero) or not.

## Examples

``` r
if (torch_is_installed()) {

a <- torch_tensor(c(0.7, -1.2, 0., 2.3))
torch_signbit(a)
}
```
