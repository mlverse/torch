# Topk

Topk

## Usage

``` r
torch_topk(self, k, dim = -1L, largest = TRUE, sorted = TRUE)
```

## Arguments

- self:

  (Tensor) the input tensor.

- k:

  (int) the k in "top-k"

- dim:

  (int, optional) the dimension to sort along

- largest:

  (bool, optional) controls whether to return largest or smallest
  elements

- sorted:

  (bool, optional) controls whether to return the elements in sorted
  order

## topk(input, k, dim=NULL, largest=TRUE, sorted=TRUE) -\> (Tensor, LongTensor)

Returns the `k` largest elements of the given `input` tensor along a
given dimension.

If `dim` is not given, the last dimension of the `input` is chosen.

If `largest` is `FALSE` then the `k` smallest elements are returned.

A namedtuple of `(values, indices)` is returned, where the `indices` are
the indices of the elements in the original `input` tensor.

The boolean option `sorted` if `TRUE`, will make sure that the returned
`k` elements are themselves sorted

## Examples

``` r
if (torch_is_installed()) {

x = torch_arange(1., 6.)
x
torch_topk(x, 3)
}
#> [[1]]
#> torch_tensor
#>  6
#>  5
#>  4
#> [ CPUFloatType{3} ]
#> 
#> [[2]]
#> torch_tensor
#>  6
#>  5
#>  4
#> [ CPULongType{3} ]
#> 
```
