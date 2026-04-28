# Flatten

Flatten

## Usage

``` r
torch_flatten(self, dims, start_dim = 1L, end_dim = -1L, out_dim)
```

## Arguments

- self:

  (Tensor) the input tensor.

- dims:

  if tensor is named you can pass the name of the dimensions to flatten

- start_dim:

  (int) the first dim to flatten

- end_dim:

  (int) the last dim to flatten

- out_dim:

  the name of the resulting dimension if a named tensor.

## flatten(input, start_dim=0, end_dim=-1) -\> Tensor

Flattens a contiguous range of dims in a tensor.

## Examples

``` r
if (torch_is_installed()) {

t = torch_tensor(matrix(c(1, 2), ncol = 2))
torch_flatten(t)
torch_flatten(t, start_dim=2)
}
#> torch_tensor
#>  1  2
#> [ CPUFloatType{1,2} ]
```
