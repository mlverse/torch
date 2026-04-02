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
#>  0.4208 -1.3835 -0.3495 -0.8847
#>  0.6759 -0.7865  0.2161 -0.7153
#>  0.8629  2.4899  0.8086  1.4238
#> [ CPUFloatType{3,4} ]
#> 
#> [[2]]
#> torch_tensor
#>  2  1  2  2
#>  3  3  1  3
#>  1  2  3  1
#> [ CPULongType{3,4} ]
#> 
```
