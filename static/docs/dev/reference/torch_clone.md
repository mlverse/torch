# Clone

Clone

## Usage

``` r
torch_clone(self, memory_format = NULL)
```

## Arguments

- self:

  (Tensor) the input tensor.

- memory_format:

  a torch memory format. see
  [`torch_preserve_format()`](https://torch.mlverse.org/docs/dev/reference/torch_memory_format.md).

## Note

This function is differentiable, so gradients will flow back from the
result of this operation to `input`. To create a tensor without an
autograd relationship to `input` see `Tensor$detach`.

## clone(input, \*, memory_format=torch.preserve_format) -\> Tensor

Returns a copy of `input`.
