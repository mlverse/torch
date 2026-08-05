# Square

Square

## Usage

``` r
torch_square(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## square(input, out=NULL) -\> Tensor

Returns a new tensor with the square of the elements of `input`.

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_square(a)
}
#> torch_tensor
#>  0.1657
#>  2.2907
#>  0.1402
#>  0.1189
#> [ CPUFloatType{4} ]
```
