# Softmin

Applies a softmin function.

## Usage

``` r
nnf_softmin(input, dim, dtype = NULL)
```

## Arguments

- input:

  (Tensor) input

- dim:

  (int) A dimension along which softmin will be computed (so every slice
  along dim will sum to 1).

- dtype:

  (`torch.dtype`, optional) the desired data type of returned tensor. If
  specified, the input tensor is casted to `dtype` before the operation
  is performed. This is useful for preventing data type overflows.
  Default: NULL.

## Details

Note that

\$\$Softmin(x) = Softmax(-x)\$\$.

See
[nnf_softmax](https://torch.mlverse.org/docs/dev/reference/nnf_softmax.md)
definition for mathematical formula.
