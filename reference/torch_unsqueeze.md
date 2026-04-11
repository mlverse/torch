# Unsqueeze

Unsqueeze

## Usage

``` r
torch_unsqueeze(self, dim)
```

## Arguments

- self:

  (Tensor) the input tensor.

- dim:

  (int) the index at which to insert the singleton dimension

## unsqueeze(input, dim) -\> Tensor

Returns a new tensor with a dimension of size one inserted at the
specified position.

The returned tensor shares the same underlying data with this tensor.

A `dim` value within the range `[-input.dim() - 1, input.dim() + 1)` can
be used. Negative `dim` will correspond to `unsqueeze` applied at `dim`
= `dim + input.dim() + 1`.

## Examples

``` r
if (torch_is_installed()) {

x = torch_tensor(c(1, 2, 3, 4))
torch_unsqueeze(x, 1)
torch_unsqueeze(x, 2)
}
#> torch_tensor
#>  1
#>  2
#>  3
#>  4
#> [ CPUFloatType{4,1} ]
```
