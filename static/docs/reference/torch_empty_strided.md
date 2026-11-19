# Empty_strided

Empty_strided

## Usage

``` r
torch_empty_strided(
  size,
  stride,
  dtype = NULL,
  layout = NULL,
  device = NULL,
  requires_grad = FALSE,
  pin_memory = FALSE
)
```

## Arguments

- size:

  (tuple of ints) the shape of the output tensor

- stride:

  (tuple of ints) the strides of the output tensor

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

- pin_memory:

  (bool, optional) If set, returned tensor would be allocated in the
  pinned memory. Works only for CPU tensors. Default: `FALSE`.

## empty_strided(size, stride, dtype=NULL, layout=NULL, device=NULL, requires_grad=False, pin_memory=False) -\> Tensor

Returns a tensor filled with uninitialized data. The shape and strides
of the tensor is defined by the variable argument `size` and `stride`
respectively. `torch_empty_strided(size, stride)` is equivalent to
`torch_empty(size).as_strided(size, stride)`.

## Warning

More than one element of the created tensor may refer to a single memory
location. As a result, in-place operations (especially ones that are
vectorized) may result in incorrect behavior. If you need to write to
the tensors, please clone them first.

## Examples

``` r
if (torch_is_installed()) {

a = torch_empty_strided(list(2, 3), list(1, 2))
a
a$stride(1)
a$size(1)
}
#> [1] 2
```
