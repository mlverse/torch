# Unsafe_split

Unsafe_split

## Usage

``` r
torch_unsafe_split(self, split_size, dim = 1L)
```

## Arguments

- self:

  (Tensor) tensor to split.

- split_size:

  (int) size of a single chunk or list of sizes for each chunk

- dim:

  (int) dimension along which to split the tensor.

## unsafe_split(tensor, split_size_or_sections, dim=0) -\> List of Tensors

Works like
[`torch_split()`](https://torch.mlverse.org/docs/dev/reference/torch_split.md)
but without enforcing the autograd restrictions on inplace modification
of the outputs.

## Warning

This function is safe to use as long as only the input, or only the
outputs are modified inplace after calling this function. It is user's
responsibility to ensure that is the case. If both the input and one or
more of the outputs are modified inplace, gradients computed by autograd
will be silently incorrect.
