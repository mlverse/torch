# Transformer Encoder Layer Module (R torch)

Implements a single transformer encoder layer as in PyTorch, including
self-attention, feed-forward network, residual connections, and layer
normalization.

## Usage

``` r
nn_transformer_encoder_layer(
  d_model,
  nhead,
  dim_feedforward = 2048,
  dropout = 0.1,
  activation = "relu",
  layer_norm_eps = 1e-05,
  batch_first = FALSE,
  norm_first = FALSE,
  bias = TRUE
)
```

## Arguments

- d_model:

  (integer) the number of expected features in the input.

- nhead:

  (integer) the number of heads in the multihead attention.

- dim_feedforward:

  (integer) the dimension of the feed-forward network model. Default:
  2048.

- dropout:

  (numeric) the dropout probability for both attention and feed-forward
  networks. Default: 0.1.

- activation:

  (character or function) the activation function of the intermediate
  layer. Can be `"relu"` or `"gelu"` or an R function mapping a tensor
  to a tensor. Default: "relu".

- layer_norm_eps:

  (numeric) the epsilon value for layer normalization. Default: 1e-5.

- batch_first:

  (logical) if TRUE, inputs are (batch, seq, feature) instead of (seq,
  batch, feature). Default: FALSE.

- norm_first:

  (logical) if TRUE, layer norm is applied before attention and
  feed-forward sublayers (Pre-Norm); if FALSE, applied after
  (Post-Norm). Default: FALSE.

- bias:

  (logical) if FALSE, the Linear layers and LayerNorm will not learn
  additive bias (LayerNorm will have no affine params). Default: TRUE.

## Value

An `nn_module` object of class `nn_transformer_encoder_layer`. Calling
the module on an input tensor returns the output tensor of the same
shape.

## Details

This module is equivalent to `torch::nn_transformer_encoder_layer` in
PyTorch, with identical default arguments and behavior. It consists of a
multi-head self-attention (`self_attn`), followed by a feed-forward
network (two linear layers with an activation in between), each part
having residual connections and dropout. Two LayerNorm layers (`norm1`,
`norm2`) are used either in pre-norm or post-norm fashion based on
`norm_first`.

The `forward()` method supports an optional `src_mask` (attention mask)
and `src_key_padding_mask` to mask out positions, and an `is_causal`
flag for auto-regressive masking. If `is_causal=TRUE`, a causal mask
will be applied (equivalent to a lower-triangular attention mask), which
should not be combined with an explicit `src_mask`.

## Examples

``` r
if (torch_is_installed()) {
if (torch_is_installed()) {
  layer <- nn_transformer_encoder_layer(d_model = 16, nhead = 4)
  x <- torch_randn(10, 2, 16) # (sequence, batch, features)
  y <- layer(x) # output has shape (10, 2, 16)
}
}
```
