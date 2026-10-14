# Range

Range

## Usage

``` r
torch_range(
  start,
  end,
  step = 1,
  dtype = NULL,
  layout = NULL,
  device = NULL,
  requires_grad = FALSE
)
```

## Arguments

- start:

  (float) the starting value for the set of points. Default: `0`.

- end:

  (float) the ending value for the set of points

- step:

  (float) the gap between each pair of adjacent points. Default: `1`.

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

## range(start=0, end, step=1, out=NULL, dtype=NULL, layout=torch.strided, device=NULL, requires_grad=False) -\> Tensor

Returns a 1-D tensor of size \\\left\lfloor \frac{\mbox{end} -
\mbox{start}}{\mbox{step}} \right\rfloor + 1\\ with values from `start`
to `end` with step `step`. Step is the gap between two values in the
tensor.

\$\$ \mbox{out}\_{i+1} = \mbox{out}\_i + \mbox{step}. \$\$

## Warning

This function is deprecated in favor of
[`torch_arange`](https://torch.mlverse.org/docs/dev/reference/torch_arange.md).

## Examples

``` r
if (torch_is_installed()) {

torch_range(1, 4)
torch_range(1, 4, 0.5)
}
#> Warning: This function is deprecated in favor of torch_arange.
#> Warning: This function is deprecated in favor of torch_arange.
#> torch_tensor
#>  1.0000
#>  1.5000
#>  2.0000
#>  2.5000
#>  3.0000
#>  3.5000
#>  4.0000
#> [ CPUFloatType{7} ]
```
