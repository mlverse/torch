# Eye

Eye

## Usage

``` r
torch_eye(
  n,
  m = n,
  dtype = NULL,
  layout = NULL,
  device = NULL,
  requires_grad = FALSE
)
```

## Arguments

- n:

  (int) the number of rows

- m:

  (int, optional) the number of columns with default being `n`

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

## eye(n, m=NULL, out=NULL, dtype=NULL, layout=torch.strided, device=NULL, requires_grad=False) -\> Tensor

Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.

## Examples

``` r
if (torch_is_installed()) {

torch_eye(3)
}
#> torch_tensor
#>  1  0  0
#>  0  1  0
#>  0  0  1
#> [ CPUFloatType{3,3} ]
```
