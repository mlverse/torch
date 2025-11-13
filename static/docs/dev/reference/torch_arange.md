# Arange

Arange

## Usage

``` r
torch_arange(
  start,
  end,
  step = 1L,
  dtype = NULL,
  layout = NULL,
  device = NULL,
  requires_grad = FALSE
)
```

## Arguments

- start:

  (Number) the starting value for the set of points. Default: `0`.

- end:

  (Number) the ending value for the set of points

- step:

  (Number) the gap between each pair of adjacent points. Default: `1`.

- dtype:

  (`torch.dtype`, optional) the desired data type of returned tensor.
  Default: if `NULL`, uses a global default (see
  `torch_set_default_tensor_type`). If `dtype` is not given, infer the
  data type from the other input arguments. If any of `start`, `end`, or
  `stop` are floating-point, the `dtype` is inferred to be the default
  dtype, see `~torch.get_default_dtype`. Otherwise, the `dtype` is
  inferred to be `torch.int64`.

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

## arange(start=0, end, step=1, out=NULL, dtype=NULL, layout=torch.strided, device=NULL, requires_grad=False) -\> Tensor

Returns a 1-D tensor of size \\\left\lceil \frac{\mbox{end} -
\mbox{start}}{\mbox{step}} \right\rceil\\ with values from the interval
`[start, end)` taken with common difference `step` beginning from
`start`.

Note that non-integer `step` is subject to floating point rounding
errors when comparing against `end`; to avoid inconsistency, we advise
adding a small epsilon to `end` in such cases.

\$\$ \mbox{out}\_{{i+1}} = \mbox{out}\_{i} + \mbox{step} \$\$

## Examples

``` r
if (torch_is_installed()) {

torch_arange(start = 0, end = 5)
torch_arange(1, 4)
torch_arange(1, 2.5, 0.5)
}
#> torch_tensor
#>  1.0000
#>  1.5000
#>  2.0000
#>  2.5000
#> [ CPUFloatType{4} ]
```
