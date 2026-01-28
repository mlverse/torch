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
#> -0.5536 -1.7158 -5.7793
#>  0.4371 -0.2878  0.4335
#>  0.2697  0.5540  1.3554
#> -0.6579  0.2375 -0.2250
#> [ CPUFloatType{4,3} ]
```
