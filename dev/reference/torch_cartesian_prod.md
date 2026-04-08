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
```
