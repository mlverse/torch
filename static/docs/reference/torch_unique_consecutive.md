# Unique_consecutive

Unique_consecutive

## Usage

``` r
torch_unique_consecutive(
  self,
  return_inverse = FALSE,
  return_counts = FALSE,
  dim = NULL
)
```

## Arguments

- self:

  (Tensor) the input tensor

- return_inverse:

  (bool) Whether to also return the indices for where elements in the
  original input ended up in the returned unique list.

- return_counts:

  (bool) Whether to also return the counts for each unique element.

- dim:

  (int) the dimension to apply unique. If `NULL`, the unique of the
  flattened input is returned. default: `NULL`

## TEST

Eliminates all but the first element from every consecutive group of
equivalent elements.

    .. note:: This function is different from [`torch_unique`] in the sense that this function
        only eliminates consecutive duplicate values. This semantics is similar to `std::unique`
        in C++.

## Examples

``` r
if (torch_is_installed()) {
x = torch_tensor(c(1, 1, 2, 2, 3, 1, 1, 2))
output = torch_unique_consecutive(x)
output
torch_unique_consecutive(x, return_inverse=TRUE)
torch_unique_consecutive(x, return_counts=TRUE)
}
#> [[1]]
#> torch_tensor
#>  1
#>  2
#>  3
#>  1
#>  2
#> [ CPUFloatType{5} ]
#> 
#> [[2]]
#> torch_tensor
#> [ CPULongType{0} ]
#> 
#> [[3]]
#> torch_tensor
#>  2
#>  2
#>  1
#>  2
#>  1
#> [ CPULongType{5} ]
#> 
```
