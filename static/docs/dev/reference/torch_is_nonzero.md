# Is_nonzero

Is_nonzero

## Usage

``` r
torch_is_nonzero(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## is_nonzero(input) -\> (bool)

Returns TRUE if the `input` is a single element tensor which is not
equal to zero after type conversions. i.e. not equal to
`torch_tensor(c(0))` or `torch_tensor(c(0))` or
`torch_tensor(c(FALSE))`. Throws a `RuntimeError` if
`torch_numel() != 1` (even in case of sparse tensors).

## Examples

``` r
if (torch_is_installed()) {

torch_is_nonzero(torch_tensor(c(0.)))
torch_is_nonzero(torch_tensor(c(1.5)))
torch_is_nonzero(torch_tensor(c(FALSE)))
torch_is_nonzero(torch_tensor(c(3)))
if (FALSE) {
torch_is_nonzero(torch_tensor(c(1, 3, 5)))
torch_is_nonzero(torch_tensor(c()))
}
}
```
