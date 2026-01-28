# Rot90

Rot90

## Usage

``` r
torch_rot90(self, k = 1L, dims = c(0, 1))
```

## Arguments

- self:

  (Tensor) the input tensor.

- k:

  (int) number of times to rotate

- dims:

  (a list or tuple) axis to rotate

## rot90(input, k, dims) -\> Tensor

Rotate a n-D tensor by 90 degrees in the plane specified by dims axis.
Rotation direction is from the first towards the second axis if k \> 0,
and from the second towards the first for k \< 0.

## Examples

``` r
if (torch_is_installed()) {

x <- torch_arange(1, 4)$view(c(2, 2))
x
torch_rot90(x, 1, c(1, 2))
x <- torch_arange(1, 8)$view(c(2, 2, 2))
x
torch_rot90(x, 1, c(1, 2))
}
#> torch_tensor
#> (1,.,.) = 
#>   3  4
#>   7  8
#> 
#> (2,.,.) = 
#>   1  2
#>   5  6
#> [ CPUFloatType{2,2,2} ]
```
