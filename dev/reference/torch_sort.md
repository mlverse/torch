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
#> -0.7806 -0.4691 -1.4752 -0.4763
#> -0.4187  0.4619  0.1924 -0.4591
#>  0.4508  3.2833  0.2304 -0.1575
#> [ CPUFloatType{3,4} ]
#> 
#> [[2]]
#> torch_tensor
#>  2  3  1  2
#>  3  1  2  3
#>  1  2  3  1
#> [ CPULongType{3,4} ]
#> 
```
