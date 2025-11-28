# Argsort

Argsort

## Usage

``` r
torch_argsort(self, dim = -1L, descending = FALSE)
```

## Arguments

- self:

  (Tensor) the input tensor.

- dim:

  (int, optional) the dimension to sort along

- descending:

  (bool, optional) controls the sorting order (ascending or descending)

## argsort(input, dim=-1, descending=False) -\> LongTensor

Returns the indices that sort a tensor along a given dimension in
ascending order by value.

This is the second value returned by `torch_sort`. See its documentation
for the exact semantics of this method.

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(4, 4))
a
torch_argsort(a, dim=1)
}
#> torch_tensor
#>  1  1  2  4
#>  4  4  3  2
#>  3  3  1  3
#>  2  2  4  1
#> [ CPULongType{4,4} ]
```
