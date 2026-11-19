# Ones_like

Ones_like

## Usage

``` r
torch_ones_like(
  input,
  dtype = NULL,
  layout = NULL,
  device = NULL,
  requires_grad = FALSE,
  memory_format = torch_preserve_format()
)
```

## Arguments

- input:

  (Tensor) the size of `input` will determine size of the output tensor.

- dtype:

  (`torch.dtype`, optional) the desired data type of returned Tensor.
  Default: if `NULL`, defaults to the dtype of `input`.

- layout:

  (`torch.layout`, optional) the desired layout of returned tensor.
  Default: if `NULL`, defaults to the layout of `input`.

- device:

  (`torch.device`, optional) the desired device of returned tensor.
  Default: if `NULL`, defaults to the device of `input`.

- requires_grad:

  (bool, optional) If autograd should record operations on the returned
  tensor. Default: `FALSE`.

- memory_format:

  (`torch.memory_format`, optional) the desired memory format of
  returned Tensor. Default: `torch_preserve_format`.

## ones_like(input, dtype=NULL, layout=NULL, device=NULL, requires_grad=False, memory_format=torch.preserve_format) -\> Tensor

Returns a tensor filled with the scalar value `1`, with the same size as
`input`. `torch_ones_like(input)` is equivalent to
`torch_ones(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)`.

## Warning

As of 0.4, this function does not support an `out` keyword. As an
alternative, the old `torch_ones_like(input, out=output)` is equivalent
to `torch_ones(input.size(), out=output)`.

## Examples

``` r
if (torch_is_installed()) {

input = torch_empty(c(2, 3))
torch_ones_like(input)
}
#> torch_tensor
#>  1  1  1
#>  1  1  1
#> [ CPUFloatType{2,3} ]
```
