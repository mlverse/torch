# Orgqr

Orgqr

## Usage

``` r
torch_orgqr(self, input2)
```

## Arguments

- self:

  (Tensor) the `a` from
  [`torch_geqrf`](https://torch.mlverse.org/docs/dev/reference/torch_geqrf.md).

- input2:

  (Tensor) the `tau` from
  [`torch_geqrf`](https://torch.mlverse.org/docs/dev/reference/torch_geqrf.md).

## orgqr(input, input2) -\> Tensor

Computes the orthogonal matrix `Q` of a QR factorization, from the
`(input, input2)` tuple returned by
[`torch_geqrf`](https://torch.mlverse.org/docs/dev/reference/torch_geqrf.md).

This directly calls the underlying LAPACK function `?orgqr`. See
`LAPACK documentation for orgqr`\_ for further details.
