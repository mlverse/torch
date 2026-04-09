# Deg2rad

Deg2rad

## Usage

``` r
torch_deg2rad(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## deg2rad(input, \*, out=None) -\> Tensor

Returns a new tensor with each of the elements of `input` converted from
angles in degrees to radians.

## Examples

``` r
if (torch_is_installed()) {

a <- torch_tensor(rbind(c(180.0, -180.0), c(360.0, -360.0), c(90.0, -90.0)))
torch_deg2rad(a)
}
#> torch_tensor
#>  3.1416 -3.1416
#>  6.2832 -6.2832
#>  1.5708 -1.5708
#> [ CPUFloatType{3,2} ]
```
