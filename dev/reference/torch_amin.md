# Amin

Amin

## Usage

``` r
torch_amin(self, dim = list(), keepdim = FALSE)
```

## Arguments

- self:

  (Tensor) the input tensor.

- dim:

  (int or tuple of ints) the dimension or dimensions to reduce.

- keepdim:

  (bool) whether the output tensor has `dim` retained or not.

## Note

The difference between `max`/`min` and `amax`/`amin` is:

- `amax`/`amin` supports reducing on multiple dimensions,

- `amax`/`amin` does not return indices,

- `amax`/`amin` evenly distributes gradient between equal values, while
  `max(dim)`/`min(dim)` propagates gradient only to a single index in
  the source tensor.

If `keepdim` is `TRUE`, the output tensors are of the same size as
`input` except in the dimension(s) `dim` where they are of size 1.
Otherwise, `dim`s are squeezed (see
[`torch_squeeze()`](https://torch.mlverse.org/docs/dev/reference/torch_squeeze.md)),
resulting in the output tensors having fewer dimensions than `input`.

## amin(input, dim, keepdim=FALSE, \*, out=None) -\> Tensor

Returns the minimum value of each slice of the `input` tensor in the
given dimension(s) `dim`.

## Examples

``` r
if (torch_is_installed()) {

a <- torch_randn(c(4, 4))
a
torch_amin(a, 1)
}
#> torch_tensor
#> -0.9475
#> -0.9782
#> -0.1537
#>  0.2831
#> [ CPUFloatType{4} ]
```
