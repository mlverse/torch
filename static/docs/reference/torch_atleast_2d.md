# Atleast_2d

Returns a 2-dimensional view of each each input tensor with zero
dimensions. Input tensors with two or more dimensions are returned
as-is.

## Usage

``` r
torch_atleast_2d(self)
```

## Arguments

- self:

  (Tensor or list of Tensors)

## Examples

``` r
if (torch_is_installed()) {

x <- torch_tensor(1.)
x
torch_atleast_2d(x)
x <- torch_randn(c(2,2))
x
torch_atleast_2d(x)
x <- torch_tensor(0.5)
y <- torch_tensor(1.)
torch_atleast_2d(list(x,y))
}
#> [[1]]
#> torch_tensor
#>  0.5000
#> [ CPUFloatType{1,1} ]
#> 
#> [[2]]
#> torch_tensor
#>  1
#> [ CPUFloatType{1,1} ]
#> 
```
