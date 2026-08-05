# MultiHead attention

Allows the model to jointly attend to information from different
representation subspaces. See reference: Attention Is All You Need

## Usage

``` r
nn_multihead_attention(
  embed_dim,
  num_heads,
  dropout = 0,
  bias = TRUE,
  add_bias_kv = FALSE,
  add_zero_attn = FALSE,
  kdim = NULL,
  vdim = NULL,
  batch_first = FALSE
)
```

## Arguments

- embed_dim:

  total dimension of the model.

- num_heads:

  parallel attention heads. Note that `embed_dim` will be split across
  `num_heads` (i.e. each head will have dimension
  `embed_dim %/% num_heads`).

- dropout:

  a Dropout layer on attn_output_weights. Default: 0.0.

- bias:

  add bias as module parameter. Default: True.

- add_bias_kv:

  add bias to the key and value sequences at dim=0.

- add_zero_attn:

  add a new batch of zeros to the key and value sequences at dim=1.

- kdim:

  total number of features in key. Default: `NULL`

- vdim:

  total number of features in value. Default: `NULL`. Note: if kdim and
  vdim are `NULL`, they will be set to embed_dim such that query, key,
  and value have the same number of features.

- batch_first:

  if `TRUE` then the input and output tensors are \\(N, S, E)\\ instead
  of \\(S, N, E)\\, where N is the batch size, S is the sequence length,
  and E is the embedding dimension.

## Details

\$\$ \mbox{MultiHead}(Q, K, V) = \mbox{Concat}(head_1,\dots,head_h)W^O
\mbox{where} head_i = \mbox{Attention}(QW_i^Q, KW_i^K, VW_i^V) \$\$

## Shape

Inputs:

- query: \\(L, N, E)\\ where L is the target sequence length, N is the
  batch size, E is the embedding dimension. (but see the `batch_first`
  argument)

- key: \\(S, N, E)\\, where S is the source sequence length, N is the
  batch size, E is the embedding dimension. (but see the `batch_first`
  argument)

- value: \\(S, N, E)\\ where S is the source sequence length, N is the
  batch size, E is the embedding dimension. (but see the `batch_first`
  argument)

- key_padding_mask: \\(N, S)\\ where N is the batch size, S is the
  source sequence length. If a ByteTensor is provided, the non-zero
  positions will be ignored while the position with the zero positions
  will be unchanged. If a BoolTensor is provided, the positions with the
  value of `True` will be ignored while the position with the value of
  `False` will be unchanged.

- attn_mask: 2D mask \\(L, S)\\ where L is the target sequence length, S
  is the source sequence length. 3D mask \\(N\*num_heads, L, S)\\ where
  N is the batch size, L is the target sequence length, S is the source
  sequence length. attn_mask ensure that position i is allowed to attend
  the unmasked positions. If a ByteTensor is provided, the non-zero
  positions are not allowed to attend while the zero positions will be
  unchanged. If a BoolTensor is provided, positions with `True` are not
  allowed to attend while `False` values will be unchanged. If a
  FloatTensor is provided, it will be added to the attention weight.

Outputs:

- attn_output: \\(L, N, E)\\ where L is the target sequence length, N is
  the batch size, E is the embedding dimension. (but see the
  `batch_first` argument)

- attn_output_weights:

  - if `avg_weights` is `TRUE` (the default), the output attention
    weights are averaged over the attention heads, giving a tensor of
    shape \\(N, L, S)\\ where N is the batch size, L is the target
    sequence length, S is the source sequence length.

  - if `avg_weights` is `FALSE`, the attention weight tensor is output
    as-is, with shape \\(N, H, L, S)\\, where H is the number of
    attention heads.

## Examples

``` r
if (torch_is_installed()) {
if (FALSE) { # \dontrun{
multihead_attn <- nn_multihead_attention(embed_dim, num_heads)
out <- multihead_attn(query, key, value)
attn_output <- out[[1]]
attn_output_weights <- out[[2]]
} # }

}
```
