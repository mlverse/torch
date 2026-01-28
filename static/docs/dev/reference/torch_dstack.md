# Dstack

Dstack

## Usage

``` r
torch_dstack(tensors)
```

## Arguments

- tensors:

  (sequence of Tensors) sequence of tensors to concatenate

## dstack(tensors, \*, out=None) -\> Tensor

Stack tensors in sequence depthwise (along third axis).

This is equivalent to concatenation along the third axis after 1-D and
2-D tensors have been reshaped by
[`torch_atleast_3d()`](https://torch.mlverse.org/docs/dev/reference/torch_atleast_3d.md).

## Examples

``` r
if (torch_is_installed()) {

a <- torch_tensor(c(1, 2, 3))
b <- torch_tensor(c(4, 5, 6))
torch_dstack(list(a,b))
a <- torch_tensor(rbind(1,2,3))
b <- torch_tensor(rbind(4,5,6))
torch_dstack(list(a,b))
}
#> torch_tensor
#> (1,.,.) = 
#>   1  4
#> 
#> (2,.,.) = 
#>   2  5
#> 
#> (3,.,.) = 
#>   3  6
#> [ CPUFloatType{3,1,2} ]
```
