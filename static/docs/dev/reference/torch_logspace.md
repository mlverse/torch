# Logspace

Logspace

## Usage

``` r
torch_logspace(
  start,
  end,
  steps = 100,
  base = 10,
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

- base:

  (float) base of the logarithm function. Default: `10.0`.

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

## logspace(start, end, steps=100, base=10.0, out=NULL, dtype=NULL, layout=torch.strided, device=NULL, requires_grad=False) -\> Tensor

Returns a one-dimensional tensor of `steps` points logarithmically
spaced with base `base` between \\{\mbox{base}}^{\mbox{start}}\\ and
\\{\mbox{base}}^{\mbox{end}}\\.

The output tensor is 1-D of size `steps`.

## Examples

``` r
if (torch_is_installed()) {

torch_logspace(start=-10, end=10, steps=5)
torch_logspace(start=0.1, end=1.0, steps=5)
torch_logspace(start=0.1, end=1.0, steps=1)
torch_logspace(start=2, end=2, steps=1, base=2)
}
#> torch_tensor
#>  4
#> [ CPUFloatType{1} ]
```
