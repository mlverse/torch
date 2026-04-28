# Trace

Trace

## Usage

``` r
torch_trace(self)
```

## Arguments

- self:

  the input tensor

## trace(input) -\> Tensor

Returns the sum of the elements of the diagonal of the input 2-D matrix.

## Examples

``` r
if (torch_is_installed()) {

x <- torch_arange(1, 9)$view(c(3, 3))
x
torch_trace(x)
}
#> torch_tensor
#> 15
#> [ CPUFloatType{} ]
```
