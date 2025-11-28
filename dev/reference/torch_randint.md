# Randint

Randint

## Usage

``` r
torch_randint(
  low,
  high,
  size,
  generator = NULL,
  dtype = NULL,
  layout = NULL,
  device = NULL,
  requires_grad = FALSE,
  memory_format = torch_preserve_format()
)
```

## Arguments

- low:

  (int, optional) Lowest integer to be drawn from the distribution.
  Default: 0.

- high:

  (int) One above the highest integer to be drawn from the distribution.

- size:

  (tuple) a tuple defining the shape of the output tensor.

- generator:

  (`torch.Generator`, optional) a pseudorandom number generator for
  sampling

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

- memory_format:

  memory format for the resulting tensor.

## randint(low=0, high, size, \*, generator=NULL, out=NULL, \\

dtype=NULL, layout=torch.strided, device=NULL, requires_grad=False) -\>
Tensor

Returns a tensor filled with random integers generated uniformly between
`low` (inclusive) and `high` (exclusive).

The shape of the tensor is defined by the variable argument `size`.

.. note: With the global dtype default (`torch_float32`), this function
returns a tensor with dtype `torch_int64`.

## Examples

``` r
if (torch_is_installed()) {

torch_randint(3, 5, list(3))
torch_randint(0, 10, size = list(2, 2))
torch_randint(3, 10, list(2, 2))
}
#> torch_tensor
#>  5  5
#>  8  3
#> [ CPUFloatType{2,2} ]
```
