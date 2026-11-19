# T

T

## Usage

``` r
torch_t(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## t(input) -\> Tensor

Expects `input` to be \<= 2-D tensor and transposes dimensions 0 and 1.

0-D and 1-D tensors are returned as is. When input is a 2-D tensor this
is equivalent to `transpose(input, 0, 1)`.

## Examples

``` r
if (torch_is_installed()) {

x = torch_randn(c(2,3))
x
torch_t(x)
x = torch_randn(c(3))
x
torch_t(x)
x = torch_randn(c(2, 3))
x
torch_t(x)
}
#> torch_tensor
#>  0.3024 -0.7289
#> -0.4378 -0.6004
#> -0.0572  0.2469
#> [ CPUFloatType{3,2} ]
```
