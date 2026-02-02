# Randint_like

Randint_like

## Usage

``` r
torch_randint_like(
  input,
  low,
  high,
  dtype = NULL,
  layout = NULL,
  device = NULL,
  requires_grad = FALSE
)
```

## Arguments

- input:

  (Tensor) the size of `input` will determine size of the output tensor.

- low:

  (int, optional) Lowest integer to be drawn from the distribution.
  Default: 0.

- high:

  (int) One above the highest integer to be drawn from the distribution.

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

## randint_like(input, low=0, high, dtype=NULL, layout=torch.strided, device=NULL, requires_grad=False,

memory_format=torch.preserve_format) -\> Tensor

Returns a tensor with the same shape as Tensor `input` filled with
random integers generated uniformly between `low` (inclusive) and `high`
(exclusive).

.. note: With the global dtype default (`torch_float32`), this function
returns a tensor with dtype `torch_int64`.
