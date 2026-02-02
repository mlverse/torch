# Multi head attention forward

Allows the model to jointly attend to information from different
representation subspaces. See reference: Attention Is All You Need

## Usage

``` r
nnf_multi_head_attention_forward(
  query,
  key,
  value,
  embed_dim_to_check,
  num_heads,
  in_proj_weight,
  in_proj_bias,
  bias_k,
  bias_v,
  add_zero_attn,
  dropout_p,
  out_proj_weight,
  out_proj_bias,
  training = TRUE,
  key_padding_mask = NULL,
  need_weights = TRUE,
  attn_mask = NULL,
  avg_weights = TRUE,
  use_separate_proj_weight = FALSE,
  q_proj_weight = NULL,
  k_proj_weight = NULL,
  v_proj_weight = NULL,
  static_k = NULL,
  static_v = NULL,
  batch_first = FALSE
)
```

## Arguments

- query:

  \\(L, N, E)\\ where L is the target sequence length, N is the batch
  size, E is the embedding dimension. If batch_first is TRUE, the first
  two dimensions are transposed.

- key:

  \\(S, N, E)\\, where S is the source sequence length, N is the batch
  size, E is the embedding dimension. If batch_first is TRUE, the first
  two dimensions are transposed.

- value:

  \\(S, N, E)\\ where S is the source sequence length, N is the batch
  size, E is the embedding dimension. If batch_first is TRUE, the first
  two dimensions are transposed.

- embed_dim_to_check:

  total dimension of the model.

- num_heads:

  parallel attention heads.

- in_proj_weight:

  input projection weight.

- in_proj_bias:

  input projection bias.

- bias_k:

  bias of the key and value sequences to be added at dim=0.

- bias_v:

  currently undocumented.

- add_zero_attn:

  add a new batch of zeros to the key and value sequences at dim=1.

- dropout_p:

  probability of an element to be zeroed.

- out_proj_weight:

  the output projection weight.

- out_proj_bias:

  output projection bias.

- training:

  apply dropout if is `TRUE`.

- key_padding_mask:

  \\(N, S)\\ where N is the batch size, S is the source sequence length.
  If a ByteTensor is provided, the non-zero positions will be ignored
  while the position with the zero positions will be unchanged. If a
  BoolTensor is provided, the positions with the value of `True` will be
  ignored while the position with the value of `False` will be
  unchanged.

- need_weights:

  output attn_output_weights.

- attn_mask:

  2D mask \\(L, S)\\ where L is the target sequence length, S is the
  source sequence length. 3D mask \\(N\*num_heads, L, S)\\ where N is
  the batch size, L is the target sequence length, S is the source
  sequence length. attn_mask ensure that position i is allowed to attend
  the unmasked positions. If a ByteTensor is provided, the non-zero
  positions are not allowed to attend while the zero positions will be
  unchanged. If a BoolTensor is provided, positions with `True` is not
  allowed to attend while `False` values will be unchanged. If a
  FloatTensor is provided, it will be added to the attention weight.

- avg_weights:

  Logical; whether to average attn_output_weights over the attention
  heads before outputting them. This doesn't change the returned value
  of attn_output; it only affects the returned attention weight matrix.

- use_separate_proj_weight:

  the function accept the proj. weights for query, key, and value in
  different forms. If false, in_proj_weight will be used, which is a
  combination of q_proj_weight, k_proj_weight, v_proj_weight.

- q_proj_weight:

  input projection weight and bias.

- k_proj_weight:

  currently undocumented.

- v_proj_weight:

  currently undocumented.

- static_k:

  static key and value used for attention operators.

- static_v:

  currently undocumented.

- batch_first:

  Logical; whether to expect query, key, and value to have batch as
  their first parameter, and to return output with batch first.
