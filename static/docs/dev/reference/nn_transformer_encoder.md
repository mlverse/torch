# Transformer Encoder Module (R torch)

Implements a stack of transformer encoder layers, optionally with a
final layer normalization.

## Usage

``` r
nn_transformer_encoder(encoder_layer, num_layers, norm = NULL)
```

## Arguments

- encoder_layer:

  (nn_module) an instance of `nn_transformer_encoder_layer` (or
  compatible) that defines the layer to be repeated.

- num_layers:

  (integer) the number of encoder layers to stack.

- norm:

  (nn_module or NULL) optional layer normalization module to apply after
  the last layer (e.g., `nn_layer_norm`). Default: NULL (no extra
  normalization).

## Value

An `nn_module` of class `nn_transformer_encoder`. Calling it on an input
tensor of shape `(S, N, E)` or `(N, S, E)` (depending on `batch_first`)
returns the encoded output of the same shape.

## Details

This module replicates the given `encoder_layer` `num_layers` times to
construct the Transformer encoder. If a `norm` module is provided, it
will be applied to the output of the final encoder layer. The forward
pass sequentially applies each encoder layer to the input.

## Examples

``` r
if (torch_is_installed()) {
if (torch_is_installed()) {
  layer <- nn_transformer_encoder_layer(d_model = 32, nhead = 4, batch_first = TRUE)
  model <- nn_transformer_encoder(layer, num_layers = 2)
  x <- torch_randn(8, 5, 32) # (batch, seq, feature) since batch_first=TRUE
  y <- model(x) # output shape is (8, 5, 32)
}
}
```
