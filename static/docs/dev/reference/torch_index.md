# Index torch tensors

Helper functions to index tensors.

## Usage

``` r
torch_index(self, indices)
```

## Arguments

- self:

  (Tensor) Tensor that will be indexed.

- indices:

  (`List[Tensor]`) List of indices. Indices are torch tensors with
  [`torch_long()`](https://torch.mlverse.org/docs/dev/reference/torch_dtype.md)
  dtype.
