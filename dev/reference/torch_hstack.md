# Hstack

Hstack

## Usage

``` r
torch_hstack(tensors)
```

## Arguments

- tensors:

  (sequence of Tensors) sequence of tensors to concatenate

## hstack(tensors, \*, out=None) -\> Tensor

Stack tensors in sequence horizontally (column wise).

This is equivalent to concatenation along the first axis for 1-D
tensors, and along the second axis for all other tensors.

## Examples

``` r
if (torch_is_installed()) {

a <- torch_tensor(c(1, 2, 3))
b <- torch_tensor(c(4, 5, 6))
torch_hstack(list(a,b))
a <- torch_tensor(rbind(1,2,3))
b <- torch_tensor(rbind(4,5,6))
torch_hstack(list(a,b))
}
#> torch_tensor
#>  1  4
#>  2  5
#>  3  6
#> [ CPUFloatType{3,2} ]
```
