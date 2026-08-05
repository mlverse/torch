# Softmax

Applies a softmax function.

## Usage

``` r
nnf_softmax(input, dim, dtype = NULL)
```

## Arguments

- input:

  (Tensor) input

- dim:

  (int) A dimension along which softmax will be computed.

- dtype:

  (`torch.dtype`, optional) the desired data type of returned tensor. If
  specified, the input tensor is casted to `dtype` before the operation
  is performed. This is useful for preventing data type overflows.
  Default: NULL.

## Details

Softmax is defined as:

\$\$Softmax(x\_{i}) = exp(x_i)/\sum_j exp(x_j)\$\$

It is applied to all slices along dim, and will re-scale them so that
the elements lie in the range `[0, 1]` and sum to 1.
