# Scaled Dot Product Attention

Computes scaled dot product attention on query, key and value tensors,
using an optional attention mask if passed, and applying dropout if a
probability greater than 0.0 is specified.

## Usage

``` r
torch_scaled_dot_product_attention(
  query,
  key,
  value,
  attn_mask = list(),
  dropout_p = 0L,
  is_causal = FALSE,
  scale = NULL,
  enable_gqa = FALSE
)
```

## Arguments

- query:

  (Tensor) Query tensor; shape \\(N, ..., L, E)\\.

- key:

  (Tensor) Key tensor; shape \\(N, ..., S, E)\\.

- value:

  (Tensor) Value tensor; shape \\(N, ..., S, Ev)\\.

- attn_mask:

  (Tensor, optional) Attention mask; shape must be broadcastable to the
  shape of attention weights, which is \\(N,..., L, S)\\. Two types of
  masks are supported. A boolean mask where a value of `TRUE` indicates
  that the element should take part in attention (and `FALSE` masks out
  the position). A float mask of the same type as query, key, value that
  is added to the attention score (use `-Inf` to mask out positions).
  Default: [`list()`](https://rdrr.io/r/base/list.html).

- dropout_p:

  (float) Dropout probability in the range 0.0, 1.0; if greater than
  0.0, dropout is applied during training. Default: 0.0.

- is_causal:

  (bool) If `TRUE`, assumes causal attention masking. `attn_mask` is
  ignored when `is_causal=TRUE`. Default: `FALSE`.

- scale:

  (float, optional) Scaling factor applied prior to softmax. If `NULL`,
  the default value is set to \\1/\sqrt{E}\\. Default: `NULL`.

- enable_gqa:

  (bool) If `TRUE`, enables grouped query attention (GQA) support.
  Default: `FALSE`.

## Value

A tensor with shape \\(N, ..., L, Ev)\\.

## Details

This function uses optimized fused CUDA kernels when available,
providing significant performance improvements (2-3x faster) compared to
manually computing attention. It is particularly beneficial for
transformer models.

The attention mechanism is defined as: \$\$Attention(Q, K, V) =
softmax(\frac{QK^T}{\sqrt{d_k}})V\$\$

Where \\N\\ is the batch size, \\S\\ is the source sequence length,
\\L\\ is the target sequence length, \\E\\ is the embedding dimension of
the query and key, and \\Ev\\ is the embedding dimension of the value.

The function automatically selects the best available implementation
based on hardware and input characteristics. On CUDA devices with
compatible architectures, it can use flash attention or memory-efficient
attention kernels.

## Examples

``` r
if (torch_is_installed()) {
if (torch_is_installed()) {
  # Basic usage
  query <- torch_randn(2, 8, 10, 64)  # (batch, heads, seq_len, dim)
  key <- torch_randn(2, 8, 10, 64)
  value <- torch_randn(2, 8, 10, 64)

  output <- torch_scaled_dot_product_attention(query, key, value)

  # With causal masking (for autoregressive models)
  output <- torch_scaled_dot_product_attention(
    query, key, value,
    is_causal = TRUE
  )

  # With attention mask
  seq_len <- 10
  attn_mask <- torch_ones(seq_len, seq_len)
  attn_mask <- torch_tril(attn_mask)  # Lower triangular mask
  output <- torch_scaled_dot_product_attention(
    query, key, value,
    attn_mask = attn_mask
  )
}

}
```
