# Ormqr

Ormqr

## Usage

``` r
torch_ormqr(self, input2, input3, left = TRUE, transpose = FALSE)
```

## Arguments

- self:

  (Tensor) the `a` from
  [`torch_geqrf`](https://torch.mlverse.org/docs/dev/reference/torch_geqrf.md).

- input2:

  (Tensor) the `tau` from
  [`torch_geqrf`](https://torch.mlverse.org/docs/dev/reference/torch_geqrf.md).

- input3:

  (Tensor) the matrix to be multiplied.

- left:

  see LAPACK documentation

- transpose:

  see LAPACK documentation

## ormqr(input, input2, input3, left=TRUE, transpose=False) -\> Tensor

Multiplies `mat` (given by `input3`) by the orthogonal `Q` matrix of the
QR factorization formed by
[`torch_geqrf()`](https://torch.mlverse.org/docs/dev/reference/torch_geqrf.md)
that is represented by `(a, tau)` (given by (`input`, `input2`)).

This directly calls the underlying LAPACK function `?ormqr`.
