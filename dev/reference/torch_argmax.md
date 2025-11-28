# Argmax

Argmax

## Arguments

- self:

  (Tensor) the input tensor.

- dim:

  (int) the dimension to reduce. If `NULL`, the argmax of the flattened
  input is returned.

- keepdim:

  (bool) whether the output tensor has `dim` retained or not. Ignored if
  `dim=NULL`.

## argmax(input) -\> LongTensor

Returns the indices of the maximum value of all elements in the `input`
tensor.

This is the second value returned by `torch_max`. See its documentation
for the exact semantics of this method.

## argmax(input, dim, keepdim=False) -\> LongTensor

Returns the indices of the maximum values of a tensor across a
dimension.

This is the second value returned by `torch_max`. See its documentation
for the exact semantics of this method.

## Examples

``` r
if (torch_is_installed()) {

if (FALSE) { # \dontrun{
a = torch_randn(c(4, 4))
a
torch_argmax(a)
} # }


a = torch_randn(c(4, 4))
a
torch_argmax(a, dim=1)
}
#> torch_tensor
#>  2
#>  3
#>  1
#>  4
#> [ CPULongType{4} ]
```
