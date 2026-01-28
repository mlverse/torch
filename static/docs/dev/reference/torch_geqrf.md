# Geqrf

Geqrf

## Usage

``` r
torch_geqrf(self)
```

## Arguments

- self:

  (Tensor) the input matrix

## geqrf(input, out=NULL) -\> (Tensor, Tensor)

This is a low-level function for calling LAPACK directly. This function
returns a namedtuple (a, tau) as defined in
`LAPACK documentation for geqrf`\_ .

You'll generally want to use
[`torch_qr`](https://torch.mlverse.org/docs/dev/reference/torch_qr.md)
instead.

Computes a QR decomposition of `input`, but without constructing \\Q\\
and \\R\\ as explicit separate matrices.

Rather, this directly calls the underlying LAPACK function `?geqrf`
which produces a sequence of 'elementary reflectors'.

See `LAPACK documentation for geqrf`\_ for further details.
