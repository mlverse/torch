# Take

Take

## Usage

``` r
torch_take(self, index)
```

## Arguments

- self:

  (Tensor) the input tensor.

- index:

  (LongTensor) the indices into tensor

## take(input, index) -\> Tensor

Returns a new tensor with the elements of `input` at the given indices.
The input tensor is treated as if it were viewed as a 1-D tensor. The
result takes the same shape as the indices.

## Examples

``` r
if (torch_is_installed()) {

src = torch_tensor(matrix(c(4,3,5,6,7,8), ncol = 3, byrow = TRUE))
torch_take(src, torch_tensor(c(1, 2, 5), dtype = torch_int64()))
}
#> torch_tensor
#>  4
#>  3
#>  7
#> [ CPUFloatType{3} ]
```
