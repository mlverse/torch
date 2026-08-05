# Empty_like

Empty_like

## Usage

``` r
torch_empty_like(
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

## empty_like(input, dtype=NULL, layout=NULL, device=NULL, requires_grad=False, memory_format=torch.preserve_format) -\> Tensor

Returns an uninitialized tensor with the same size as `input`.
`torch_empty_like(input)` is equivalent to
`torch_empty(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)`.

## Examples

``` r
if (torch_is_installed()) {

torch_empty(list(2,3), dtype = torch_int64())
}
#> torch_tensor
#>  0  0  0
#>  0  0  0
#> [ CPULongType{2,3} ]
```
