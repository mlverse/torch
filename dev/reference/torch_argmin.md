# Argmin

Argmin

## Arguments

- self:

  (Tensor) the input tensor.

- dim:

  (int) the dimension to reduce. If `NULL`, the argmin of the flattened
  input is returned.

- keepdim:

  (bool) whether the output tensor has `dim` retained or not. Ignored if
  `dim=NULL`.

## argmin(input) -\> LongTensor

Returns the indices of the minimum value of all elements in the `input`
tensor.

This is the second value returned by `torch_min`. See its documentation
for the exact semantics of this method.

## argmin(input, dim, keepdim=False, out=NULL) -\> LongTensor

Returns the indices of the minimum values of a tensor across a
dimension.

This is the second value returned by `torch_min`. See its documentation
for the exact semantics of this method.

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(4, 4))
a
torch_argmin(a)


a = torch_randn(c(4, 4))
a
torch_argmin(a, dim=1)
}
#> torch_tensor
#>  1
#>  4
#>  1
#>  2
#> [ CPULongType{4} ]
```
