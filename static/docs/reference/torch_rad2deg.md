# Rad2deg

Rad2deg

## Usage

``` r
torch_rad2deg(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## rad2deg(input, \*, out=None) -\> Tensor

Returns a new tensor with each of the elements of `input` converted from
angles in radians to degrees.

## Examples

``` r
if (torch_is_installed()) {

a <- torch_tensor(rbind(c(3.142, -3.142), c(6.283, -6.283), c(1.570, -1.570)))
torch_rad2deg(a)
}
#> torch_tensor
#>  180.0233 -180.0233
#>  359.9894 -359.9894
#>   89.9544  -89.9544
#> [ CPUFloatType{3,2} ]
```
