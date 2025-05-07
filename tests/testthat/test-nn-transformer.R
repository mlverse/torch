library(torch)
# Unit tests for nn_transformer_encoder_layer and nn_transformer_encoder

test_that("TransformerEncoderLayer produces output of correct shape", {
  layer <- nn_transformer_encoder_layer(d_model = 8, nhead = 2, dim_feedforward = 16)
  input <- torch_randn(5, 3, 8)  # (seq_len=5, batch=3, features=8)
  output <- layer(input)
  expect_tensor(output, min_dims = 3)
  expect_equal(dim(output), dim(input))  # output shape should match input shape
})

test_that("TransformerEncoderLayer batch_first works", {
  layer_bf <- nn_transformer_encoder_layer(d_model = 8, nhead = 2, batch_first = TRUE)
  input_bf <- torch_randn(3, 5, 8)  # (batch=3, seq_len=5, features=8)
  output_bf <- layer_bf(input_bf)
  expect_equal(dim(output_bf), dim(input_bf))
})

test_that("TransformerEncoderLayer accepts mask and is_causal", {
  d_model <- 4; seq_len <- 4; batch <- 1
  layer <- nn_transformer_encoder_layer(d_model = d_model, nhead = 1, dropout = 0)  # no dropout for deterministic result
  layer$eval()  # ensure dropout is off
  x <- torch_randn(seq_len, batch, d_model)
  # Create an explicit causal mask (upper triangular True matrix)
  causal_mask <- torch_ones(c(seq_len, seq_len), dtype = torch_bool())$triu(diagonal = 1)
  # Run with explicit mask vs using is_causal flag
  out_mask <- layer(x, src_mask = causal_mask)
  out_flag <- layer(x, is_causal = TRUE)
  expect_equal(out_mask, out_flag)  # should be identical
})

test_that("TransformerEncoderLayer key_padding_mask works as expected", {
  layer <- nn_transformer_encoder_layer(d_model = 6, nhead = 2, dropout = 0)
  layer$eval()
  seq_len <- 3; batch <- 2
  x <- torch_randn(seq_len, batch, 6)
  # key_padding_mask: mask out the last position of sequence for the second batch element
  pad_mask <- torch_tensor(matrix(c(0, 0, 0,   # no padding in batch 1
                                    0, 0, 1),  # third position masked in batch 2
                                  nrow = batch, ncol = seq_len, byrow = TRUE), dtype = torch_bool())
  out_nomask <- layer(x)
  out_masked <- layer(x, src_key_padding_mask = pad_mask)
  # The masked position output should be the same in both (because when masked, it does not attend to anything 
  # except possibly itself through residual connections)
  expect_equal(out_nomask[ ,2, ][3, ], out_masked[ ,2, ][3, ])
  # (Here we compare the last time-step of batch 2 from both outputs.)
})

test_that("TransformerEncoder (stack of layers) produces correct output and uses each layer", {
  # Create a base layer and an encoder with multiple layers
  base_layer <- nn_transformer_encoder_layer(d_model = 8, nhead = 2, dropout = 0)
  model <- nn_transformer_encoder(base_layer, num_layers = 3)
  model$eval()
  x <- torch_randn(4, 2, 8)  # (seq_len=4, batch=2, features=8)
  # Output from the encoder
  out_model <- model(x)
  expect_equal(dim(out_model), dim(x))
  # Manually apply the cloned layers sequentially and compare
  manual_out <- x
  for (i in 1:3) {
    manual_out <- model$layers[[i]](manual_out)
  }
  if (!is.null(model$norm)) {
    manual_out <- model$norm(manual_out)
  }
  expect_equal(out_model, manual_out)
})

test_that("TransformerEncoder supports different norm and preserves results", {
  # Use a final normalization in TransformerEncoder
  layer <- nn_transformer_encoder_layer(d_model = 5, nhead = 1, dropout = 0)
  final_norm <- nn_layer_norm(normalized_shape = 5)
  model <- nn_transformer_encoder(layer, num_layers = 2, norm = final_norm)
  model$eval()
  x <- torch_randn(6, 1, 5)
  out <- model(x)
  # The output with final norm should equal manually normalizing the output of encoder without norm
  model_no_norm <- nn_transformer_encoder(layer, num_layers = 2)  # same layers but without final norm
  model_no_norm$eval()
  out_no_norm <- model_no_norm(x)
  manual_norm <- final_norm(out_no_norm)
  expect_equal(out, manual_norm)
})

test_that("Different activation functions produce similar results when appropriate", {
  torch_manual_seed(42)
  layer_relu_str <- nn_transformer_encoder_layer(d_model = 4, nhead = 1, dim_feedforward = 8, 
                                                 activation = "relu", dropout = 0)
  torch_manual_seed(42)
  layer_relu_fun <- nn_transformer_encoder_layer(d_model = 4, nhead = 1, dim_feedforward = 8, 
                                                 activation = nnf_relu, dropout = 0)
  layer_relu_str$eval(); layer_relu_fun$eval()
  x <- torch_randn(5, 1, 4)
  out_str <- layer_relu_str(x)
  out_fun <- layer_relu_fun(x)
  expect_equal(out_str, out_fun)  # relu string vs function should match
  # Similarly for GELU
  torch_manual_seed(123)
  layer_gelu_str <- nn_transformer_encoder_layer(d_model = 4, nhead = 1, dim_feedforward = 8, 
                                                 activation = "gelu", dropout = 0)
  torch_manual_seed(123)
  layer_gelu_fun <- nn_transformer_encoder_layer(d_model = 4, nhead = 1, dim_feedforward = 8, 
                                                 activation = nnf_gelu, dropout = 0)
  layer_gelu_str$eval(); layer_gelu_fun$eval()
  out_str2 <- layer_gelu_str(x)
  out_fun2 <- layer_gelu_fun(x)
  expect_equal(out_str2, out_fun2)
})

test_that("Modules are serializable and gradients flow", {
  layer <- nn_transformer_encoder_layer(d_model = 3, nhead = 1)
  model <- nn_transformer_encoder(layer, num_layers = 2)
  # Serialize and deserialize
  tmp <- tempfile(fileext = ".pt")
  torch_save(model, tmp)
  model2 <- torch_load(tmp)
  # Check that loaded model outputs the same as original
  x <- torch_randn(2, 1, 3, requires_grad = TRUE)
  model$eval(); model2$eval()
  expect_equal(model(x), model2(x))
  # Gradient flow: do a backward pass
  model$train()
  out <- model(x)
  loss <- out$sum()
  loss$backward()
  # Check that at least one parameter has non-null gradient
  grads <- lapply(model$parameters, function(p) p$grad)
  has_grad <- any(sapply(grads, function(g) { !is_undefined(g) && torch_numel(g) > 0 }))
  expect_true(has_grad)
})
