# Full_like

Full_like

## Usage

``` r
torch_full_like(
  input,
  fill_value,
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

- fill_value:

  the number to fill the output tensor with.

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

## full_like(input, fill_value, out=NULL, dtype=NULL, layout=torch.strided, device=NULL, requires_grad=False,

memory_format=torch.preserve_format) -\> Tensor

Returns a tensor with the same size as `input` filled with `fill_value`.
`torch_full_like(input, fill_value)` is equivalent to
`torch_full(input.size(), fill_value, dtype=input.dtype, layout=input.layout, device=input.device)`.
