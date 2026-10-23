# Modify values selected by `indices`.

Modify values selected by `indices`.

## Usage

``` r
torch_index_put(self, indices, values, accumulate = FALSE)
```

## Arguments

- self:

  (Tensor) Tensor that will be indexed.

- indices:

  (`List[Tensor]`) List of indices. Indices are torch tensors with
  [`torch_long()`](https://torch.mlverse.org/docs/dev/reference/torch_dtype.md)
  dtype.

- values:

  (Tensor) values that will be replaced the indexed location. Used for
  `torch_index_put` and `torch_index_put_`.

- accumulate:

  (bool) Wether instead of replacing the current values with `values`,
  you want to add them.
