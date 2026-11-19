# Linspace

Linspace

## Usage

``` r
torch_linspace(
  start,
  end,
  steps = 100,
  dtype = NULL,
  layout = NULL,
  device = NULL,
  requires_grad = FALSE
)
```

## Arguments

- start:

  (float) the starting value for the set of points

- end:

  (float) the ending value for the set of points

- steps:

  (int) number of points to sample between `start` and `end`. Default:
  `100`.

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

## linspace(start, end, steps=100, out=NULL, dtype=NULL, layout=torch.strided, device=NULL, requires_grad=False) -\> Tensor

Returns a one-dimensional tensor of `steps` equally spaced points
between `start` and `end`.

The output tensor is 1-D of size `steps`.

## Examples

``` r
if (torch_is_installed()) {

torch_linspace(3, 10, steps=5)
torch_linspace(-10, 10, steps=5)
torch_linspace(start=-10, end=10, steps=5)
torch_linspace(start=-10, end=10, steps=1)
}
#> torch_tensor
#> -10
#> [ CPUFloatType{1} ]
```
