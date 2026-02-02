# Combinations

Combinations

## Usage

``` r
torch_combinations(self, r = 2L, with_replacement = FALSE)
```

## Arguments

- self:

  (Tensor) 1D vector.

- r:

  (int, optional) number of elements to combine

- with_replacement:

  (boolean, optional) whether to allow duplication in combination

## combinations(input, r=2, with_replacement=False) -\> seq

Compute combinations of length \\r\\ of the given tensor. The behavior
is similar to python's `itertools.combinations` when `with_replacement`
is set to `False`, and `itertools.combinations_with_replacement` when
`with_replacement` is set to `TRUE`.

## Examples

``` r
if (torch_is_installed()) {

a = c(1, 2, 3)
tensor_a = torch_tensor(a)
torch_combinations(tensor_a)
torch_combinations(tensor_a, r=3)
torch_combinations(tensor_a, with_replacement=TRUE)
}
#> torch_tensor
#>  1  1
#>  1  2
#>  1  3
#>  2  2
#>  2  3
#>  3  3
#> [ CPUFloatType{6,2} ]
```
