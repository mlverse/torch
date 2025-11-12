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
#>  0.1391
#>  0.0236
#>  0.8142
#>  7.3732
#> [ CPUFloatType{4} ]
```
