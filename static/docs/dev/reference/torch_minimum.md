# Minimum

Minimum

## Usage

``` r
torch_minimum(self, other)
```

## Arguments

- self:

  (Tensor) the input tensor.

- other:

  (Tensor) the second input tensor

## Note

If one of the elements being compared is a NaN, then that element is
returned. `torch_minimum()` is not supported for tensors with complex
dtypes.

## minimum(input, other, \*, out=None) -\> Tensor

Computes the element-wise minimum of `input` and `other`.

## Examples

``` r
if (torch_is_installed()) {

a <- torch_tensor(c(1, 2, -1))
b <- torch_tensor(c(3, 0, 4))
torch_minimum(a, b)
}
#> torch_tensor
#>  1
#>  0
#> -1
#> [ CPUFloatType{3} ]
```
