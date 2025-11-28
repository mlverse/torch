# Randn

Randn

## Usage

``` r
torch_randn(
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

  (int...) a sequence of integers defining the shape of the output
  tensor. Can be a variable number of arguments or a collection like a
  list or tuple.

- names:

  optional names for the dimensions

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

## randn(\*size, out=NULL, dtype=NULL, layout=torch.strided, device=NULL, requires_grad=False) -\> Tensor

Returns a tensor filled with random numbers from a normal distribution
with mean `0` and variance `1` (also called the standard normal
distribution).

\$\$ \mbox{out}\_{i} \sim \mathcal{N}(0, 1) \$\$ The shape of the tensor
is defined by the variable argument `size`.

## Examples

``` r
if (torch_is_installed()) {

torch_randn(c(4))
torch_randn(c(2, 3))
}
#> torch_tensor
#>  0.2286  1.0248  0.4478
#> -0.6994 -1.1042 -0.9320
#> [ CPUFloatType{2,3} ]
```
