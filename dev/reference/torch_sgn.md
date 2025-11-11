# Sgn

Sgn

## Usage

``` r
torch_sgn(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## sgn(input, \*, out=None) -\> Tensor

For complex tensors, this function returns a new tensor whose elemants
have the same angle as that of the elements of `input` and absolute
value 1. For a non-complex tensor, this function returns the signs of
the elements of `input` (see
[`torch_sign`](https://torch.mlverse.org/docs/dev/reference/torch_sign.md)).

\\\mbox{out}\_{i} = 0\\, if \\\|{\mbox{{input}}\_i}\| == 0\\
\\\mbox{out}\_{i} =
\frac{{\mbox{{input}}\_i}}{\|{\mbox{{input}}\_i}\|}\\, otherwise

## Examples

``` r
if (torch_is_installed()) {
if (FALSE) {
x <- torch_tensor(c(3+4i, 7-24i, 0, 1+2i))
x$sgn()
torch_sgn(x)
}
}
```
