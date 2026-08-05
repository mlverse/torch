# Meshgrid

Take \\N\\ tensors, each of which can be either scalar or 1-dimensional
vector, and create \\N\\ N-dimensional grids, where the \\i\\ `th` grid
is defined by expanding the \\i\\ `th` input over dimensions defined by
other inputs.

## Usage

``` r
torch_meshgrid(tensors, indexing)
```

## Arguments

- tensors:

  (list of Tensor) list of scalars or 1 dimensional tensors. Scalars
  will be treated (1,).

- indexing:

  (str, optional): the indexing mode, either “xy” or “ij”, defaults to
  “ij”. See warning for future changes. If “xy” is selected, the first
  dimension corresponds to the cardinality of the second input and the
  second dimension corresponds to the cardinality of the first input. If
  “ij” is selected, the dimensions are in the same order as the
  cardinality of the inputs.

## Warning

In the future `torch_meshgrid` will transition to indexing=’xy’ as the
default. This [issue](https://github.com/pytorch/pytorch/issues/50276)
tracks this issue with the goal of migrating to NumPy’s behavior.

## Examples

``` r
if (torch_is_installed()) {

x = torch_tensor(c(1, 2, 3))
y = torch_tensor(c(4, 5, 6))
out = torch_meshgrid(list(x, y))
out
}
#> [[1]]
#> torch_tensor
#>  1  1  1
#>  2  2  2
#>  3  3  3
#> [ CPUFloatType{3,3} ]
#> 
#> [[2]]
#> torch_tensor
#>  4  5  6
#>  4  5  6
#>  4  5  6
#> [ CPUFloatType{3,3} ]
#> 
```
