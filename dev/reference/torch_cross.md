# Cross

Cross

## Usage

``` r
torch_cross(self, other, dim = NULL)
```

## Arguments

- self:

  (Tensor) the input tensor.

- other:

  (Tensor) the second input tensor

- dim:

  (int, optional) the dimension to take the cross-product in.

## cross(input, other, dim=-1, out=NULL) -\> Tensor

Returns the cross product of vectors in dimension `dim` of `input` and
`other`.

`input` and `other` must have the same size, and the size of their `dim`
dimension should be 3.

If `dim` is not given, it defaults to the first dimension found with the
size 3.

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(4, 3))
a
b = torch_randn(c(4, 3))
b
torch_cross(a, b, dim=2)
torch_cross(a, b)
}
#> torch_tensor
#>  0.2666  0.4294  0.4080
#> -1.0822 -0.1342  0.1227
#>  0.6189 -1.4107 -1.1487
#> -1.1537 -0.3508  1.0745
#> [ CPUFloatType{4,3} ]
```
