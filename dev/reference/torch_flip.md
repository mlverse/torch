# Flip

Flip

## Usage

``` r
torch_flip(self, dims)
```

## Arguments

- self:

  (Tensor) the input tensor.

- dims:

  (a list or tuple) axis to flip on

## flip(input, dims) -\> Tensor

Reverse the order of a n-D tensor along given axis in dims.

## Examples

``` r
if (torch_is_installed()) {

x <- torch_arange(1, 8)$view(c(2, 2, 2))
x
torch_flip(x, c(1, 2))
}
```
