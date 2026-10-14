# Vstack

Vstack

## Usage

``` r
torch_vstack(tensors)
```

## Arguments

- tensors:

  (sequence of Tensors) sequence of tensors to concatenate

## vstack(tensors, \*, out=None) -\> Tensor

Stack tensors in sequence vertically (row wise).

This is equivalent to concatenation along the first axis after all 1-D
tensors have been reshaped by
[`torch_atleast_2d()`](https://torch.mlverse.org/docs/dev/reference/torch_atleast_2d.md).

## Examples

``` r
if (torch_is_installed()) {

a <- torch_tensor(c(1, 2, 3))
b <- torch_tensor(c(4, 5, 6))
torch_vstack(list(a,b))
a <- torch_tensor(rbind(1,2,3))
b <- torch_tensor(rbind(4,5,6))
torch_vstack(list(a,b))
}
#> torch_tensor
#>  1
#>  2
#>  3
#>  4
#>  5
#>  6
#> [ CPUFloatType{6,1} ]
```
