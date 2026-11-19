# Norm

Norm

## Usage

``` r
torch_norm(self, p = 2L, dim, keepdim = FALSE, dtype)
```

## Arguments

- self:

  (Tensor) the input tensor

- p:

  (int, float, inf, -inf, 'fro', 'nuc', optional) the order of norm.
  Default: `'fro'` The following norms can be calculated: =====
  ============================ ========================== ord matrix
  norm vector norm ===== ============================
  ========================== NULL Frobenius norm 2-norm 'fro' Frobenius
  norm – 'nuc' nuclear norm – Other as vec norm when dim is NULL
  sum(abs(x)**ord)**(1./ord) ===== ============================
  ==========================

- dim:

  (int, 2-tuple of ints, 2-list of ints, optional) If it is an int,
  vector norm will be calculated, if it is 2-tuple of ints, matrix norm
  will be calculated. If the value is NULL, matrix norm will be
  calculated when the input tensor only has two dimensions, vector norm
  will be calculated when the input tensor only has one dimension. If
  the input tensor has more than two dimensions, the vector norm will be
  applied to last dimension.

- keepdim:

  (bool, optional) whether the output tensors have `dim` retained or
  not. Ignored if `dim` = `NULL` and `out` = `NULL`. Default: `FALSE`
  Ignored if `dim` = `NULL` and `out` = `NULL`.

- dtype:

  (`torch.dtype`, optional) the desired data type of returned tensor. If
  specified, the input tensor is casted to 'dtype' while performing the
  operation. Default: NULL.

## TEST

Returns the matrix norm or vector norm of a given tensor.

## Examples

``` r
if (torch_is_installed()) {

a <- torch_arange(1, 9, dtype = torch_float())
b <- a$reshape(list(3, 3))
torch_norm(a)
torch_norm(b)
torch_norm(a, Inf)
torch_norm(b, Inf)

}
#> torch_tensor
#> 9
#> [ CPUFloatType{} ]
```
