# Converts R objects to a torch tensor

Converts R objects to a torch tensor

## Usage

``` r
torch_tensor(
  data,
  dtype = NULL,
  device = NULL,
  requires_grad = FALSE,
  pin_memory = FALSE
)
```

## Arguments

- data:

  an R atomic vector, matrix or array

- dtype:

  a
  [torch_dtype](https://torch.mlverse.org/docs/dev/reference/torch_dtype.md)
  instance

- device:

  a device creted with
  [`torch_device()`](https://torch.mlverse.org/docs/dev/reference/torch_device.md)

- requires_grad:

  if autograd should record operations on the returned tensor.

- pin_memory:

  If set, returned tensor would be allocated in the pinned memory.

## Examples

``` r
if (torch_is_installed()) {
torch_tensor(c(1, 2, 3, 4))
torch_tensor(c(1, 2, 3, 4), dtype = torch_int())
}
#> torch_tensor
#>  1
#>  2
#>  3
#>  4
#> [ CPUIntType{4} ]
```
