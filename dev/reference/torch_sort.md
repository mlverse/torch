# Sort

Sort

## Arguments

- self:

  (Tensor) the input tensor.

- dim:

  (int, optional) the dimension to sort along

- descending:

  (bool, optional) controls the sorting order (ascending or descending)

- stable:

  (bool, optional) – makes the sorting routine stable, which guarantees
  that the order of equivalent elements is preserved.

## sort(input, dim=-1, descending=FALSE) -\> (Tensor, LongTensor)

Sorts the elements of the `input` tensor along a given dimension in
ascending order by value.

If `dim` is not given, the last dimension of the `input` is chosen.

If `descending` is `TRUE` then the elements are sorted in descending
order by value.

A namedtuple of (values, indices) is returned, where the `values` are
the sorted values and `indices` are the indices of the elements in the
original `input` tensor.

## Examples

``` r
if (torch_is_installed()) {

x = torch_randn(c(3, 4))
out = torch_sort(x)
out
out = torch_sort(x, 1)
out
}
#> [[1]]
#> torch_tensor
#> -0.3363 -0.1836 -1.5590 -1.1380
#>  0.1360  0.7941  1.1101  0.0111
#>  0.7093  1.0044  1.4681  0.0445
#> [ CPUFloatType{3,4} ]
#> 
#> [[2]]
#> torch_tensor
#>  3  1  3  1
#>  2  2  1  3
#>  1  3  2  2
#> [ CPULongType{3,4} ]
#> 
```
