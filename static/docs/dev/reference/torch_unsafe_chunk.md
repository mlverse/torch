# Unsafe_chunk

Unsafe_chunk

## Usage

``` r
torch_unsafe_chunk(self, chunks, dim = 1L)
```

## Arguments

- self:

  (Tensor) the tensor to split

- chunks:

  (int) number of chunks to return

- dim:

  (int) dimension along which to split the tensor

## unsafe_chunk(input, chunks, dim=0) -\> List of Tensors

Works like
[`torch_chunk()`](https://torch.mlverse.org/docs/dev/reference/torch_chunk.md)
but without enforcing the autograd restrictions on inplace modification
of the outputs.

## Warning

This function is safe to use as long as only the input, or only the
outputs are modified inplace after calling this function. It is user's
responsibility to ensure that is the case. If both the input and one or
more of the outputs are modified inplace, gradients computed by autograd
will be silently incorrect.
