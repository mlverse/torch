# Maximum

Maximum

## Usage

``` r
torch_maximum(self, other)
```

## Arguments

- self:

  (Tensor) the input tensor.

- other:

  (Tensor) the second input tensor

## Note

If one of the elements being compared is a NaN, then that element is
returned. `torch_maximum()` is not supported for tensors with complex
dtypes.

## maximum(input, other, \*, out=None) -\> Tensor

Computes the element-wise maximum of `input` and `other`.

## Examples

``` r
if (torch_is_installed()) {

a <- torch_tensor(c(1, 2, -1))
b <- torch_tensor(c(3, 0, 4))
torch_maximum(a, b)
}
#> torch_tensor
#>  3
#>  2
#>  4
#> [ CPUFloatType{3} ]
```
