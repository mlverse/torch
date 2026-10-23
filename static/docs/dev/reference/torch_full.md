# Full

Full

## Usage

``` r
torch_full(
  size,
  fill_value,
  names = NULL,
  dtype = NULL,
  layout = NULL,
  device = NULL,
  requires_grad = FALSE
)
```

## Arguments

- size:

  (int...) a list, tuple, or `torch_Size` of integers defining the shape
  of the output tensor.

- fill_value:

  NA the number to fill the output tensor with.

- names:

  optional names of the dimensions

- dtype:

  (`torch.dtype`, optional) the desired data type of returned tensor.
  Default: if `NULL`, uses a global default (see
  `torch_set_default_tensor_type`).

- layout:

  (`torch.layout`, optional) the desired layout of returned Tensor.
  Default: `torch_strided`.

- device:

  (`torch.device`, optional) the desired device of returned tensor.
  Default: if `NULL`, uses the current device for the default tensor
  type (see `torch_set_default_tensor_type`). `device` will be the CPU
  for CPU tensor types and the current CUDA device for CUDA tensor
  types.

- requires_grad:

  (bool, optional) If autograd should record operations on the returned
  tensor. Default: `FALSE`.

## full(size, fill_value, out=NULL, dtype=NULL, layout=torch.strided, device=NULL, requires_grad=False) -\> Tensor

Returns a tensor of size `size` filled with `fill_value`.

## Warning

In PyTorch 1.5 a bool or integral `fill_value` will produce a warning if
`dtype` or `out` are not set. In a future PyTorch release, when `dtype`
and `out` are not set a bool `fill_value` will return a tensor of
torch.bool dtype, and an integral `fill_value` will return a tensor of
torch.long dtype.

## Examples

``` r
if (torch_is_installed()) {

torch_full(list(2, 3), 3.141592)
}
#> torch_tensor
#>  3.1416  3.1416  3.1416
#>  3.1416  3.1416  3.1416
#> [ CPUFloatType{2,3} ]
```
