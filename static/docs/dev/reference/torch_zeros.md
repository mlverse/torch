# Zeros

Zeros

## Usage

``` r
torch_zeros(
  ...,
  names = NULL,
  dtype = NULL,
  layout = NULL,
  device = NULL,
  requires_grad = FALSE
)
```

## Arguments

- ...:

  a sequence of integers defining the shape of the output tensor. Can be
  a variable number of arguments or a collection like a list or tuple.

- names:

  optional dimension names

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

## zeros(\*size, out=NULL, dtype=NULL, layout=torch.strided, device=NULL, requires_grad=False) -\> Tensor

Returns a tensor filled with the scalar value `0`, with the shape
defined by the variable argument `size`.

## Examples

``` r
if (torch_is_installed()) {

torch_zeros(c(2, 3))
torch_zeros(c(5))
}
#> torch_tensor
#>  0
#>  0
#>  0
#>  0
#>  0
#> [ CPUFloatType{5} ]
```
