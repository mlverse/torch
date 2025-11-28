# Trunc

Trunc

## Usage

``` r
torch_trunc(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## trunc(input, out=NULL) -\> Tensor

Returns a new tensor with the truncated integer values of the elements
of `input`.

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_trunc(a)
}
#> torch_tensor
#> -0
#>  1
#> -1
#>  0
#> [ CPUFloatType{4} ]
```
