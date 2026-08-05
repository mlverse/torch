# Cartesian_prod

Do cartesian product of the given sequence of tensors.

## Usage

``` r
torch_cartesian_prod(tensors)
```

## Arguments

- tensors:

  a list containing any number of 1 dimensional tensors.

## Examples

``` r
if (torch_is_installed()) {

a = c(1, 2, 3)
b = c(4, 5)
tensor_a = torch_tensor(a)
tensor_b = torch_tensor(b)
torch_cartesian_prod(list(tensor_a, tensor_b))
}
#> torch_tensor
#>  1  4
#>  1  5
#>  2  4
#>  2  5
#>  3  4
#>  3  5
#> [ CPUFloatType{6,2} ]
```
