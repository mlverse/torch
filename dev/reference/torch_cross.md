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
#> -1.2564  0.8650  0.0112
#>  2.0560 -0.6361 -0.3045
#> -0.1061  1.0087  2.6479
#>  0.0478 -0.1773  2.1238
#> [ CPUFloatType{4,3} ]
```
