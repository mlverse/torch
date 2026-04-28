# Randperm

Randperm

## Usage

``` r
torch_randperm(
  n,
  dtype = torch_int64(),
  layout = NULL,
  device = NULL,
  requires_grad = FALSE
)
```

## Arguments

- n:

  (int) the upper bound (exclusive)

- dtype:

  (`torch.dtype`, optional) the desired data type of returned tensor.
  Default: `torch_int64`.

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

## randperm(n, out=NULL, dtype=torch.int64, layout=torch.strided, device=NULL, requires_grad=False) -\> LongTensor

Returns a random permutation of integers from `0` to `n - 1`.

## Examples

``` r
if (torch_is_installed()) {

torch_randperm(4)
}
#> torch_tensor
#>  1
#>  2
#>  3
#>  0
#> [ CPULongType{4} ]
```
