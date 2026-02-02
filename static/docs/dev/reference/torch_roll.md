# Roll

Roll

## Usage

``` r
torch_roll(self, shifts, dims = list())
```

## Arguments

- self:

  (Tensor) the input tensor.

- shifts:

  (int or tuple of ints) The number of places by which the elements of
  the tensor are shifted. If shifts is a tuple, dims must be a tuple of
  the same size, and each dimension will be rolled by the corresponding
  value

- dims:

  (int or tuple of ints) Axis along which to roll

## roll(input, shifts, dims=NULL) -\> Tensor

Roll the tensor along the given dimension(s). Elements that are shifted
beyond the last position are re-introduced at the first position. If a
dimension is not specified, the tensor will be flattened before rolling
and then restored to the original shape.

## Examples

``` r
if (torch_is_installed()) {

x = torch_tensor(c(1, 2, 3, 4, 5, 6, 7, 8))$view(c(4, 2))
x
torch_roll(x, 1, 1)
torch_roll(x, -1, 1)
torch_roll(x, shifts=list(2, 1), dims=list(1, 2))
}
#> torch_tensor
#>  6  5
#>  8  7
#>  2  1
#>  4  3
#> [ CPUFloatType{4,2} ]
```
