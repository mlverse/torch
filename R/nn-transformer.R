#' @include nn.R
NULL


#' Transformer Encoder Layer Module (R torch)
#'
#' Implements a single transformer encoder layer as in PyTorch, including 
#' self-attention, feed-forward network, residual connections, and layer normalization.
#'
#' @param d_model (integer) the number of expected features in the input.
#' @param nhead (integer) the number of heads in the multihead attention.
#' @param dim_feedforward (integer) the dimension of the feed-forward network model. Default: 2048.
#' @param dropout (numeric) the dropout probability for both attention and feed-forward networks. Default: 0.1.
#' @param activation (character or function) the activation function of the intermediate layer. 
#'   Can be `"relu"` or `"gelu"` or an R function mapping a tensor to a tensor. Default: "relu".
#' @param layer_norm_eps (numeric) the epsilon value for layer normalization. Default: 1e-5.
#' @param batch_first (logical) if TRUE, inputs are (batch, seq, feature) instead of (seq, batch, feature). Default: FALSE.
#' @param norm_first (logical) if TRUE, layer norm is applied before attention and feed-forward sublayers (Pre-Norm); if FALSE, applied after (Post-Norm). Default: FALSE.
#' @param bias (logical) if FALSE, the Linear layers and LayerNorm will not learn additive bias (LayerNorm will have no affine params). Default: TRUE.
#'
#' @section Details:
#' This module is equivalent to `torch::nn_transformer_encoder_layer` in PyTorch, with identical default arguments and behavior. 
#' It consists of a multi-head self-attention (`self_attn`), followed by a feed-forward network (two linear layers with an activation in between), each part having residual connections and dropout. 
#' Two LayerNorm layers (`norm1`, `norm2`) are used either in pre-norm or post-norm fashion based on `norm_first`.
#'
#' The `forward()` method supports an optional `src_mask` (attention mask) and `src_key_padding_mask` to mask out positions, and an `is_causal` flag for auto-regressive masking. 
#' If `is_causal=TRUE`, a causal mask will be applied (equivalent to a lower-triangular attention mask), which should not be combined with an explicit `src_mask`.
#'
#' @returns An `nn_module` object of class `nn_transformer_encoder_layer`. Calling the module on an input tensor returns the output tensor of the same shape.
#'
#' @examples
#' if (torch_is_installed()) {
#'   layer <- nn_transformer_encoder_layer(d_model=16, nhead=4)
#'   x <- torch_randn(10, 2, 16)  # (sequence, batch, features)
#'   y <- layer(x)  # output has shape (10, 2, 16)
#' }
#' @export
nn_transformer_encoder_layer <- nn_module(
  "nn_transformer_encoder_layer",
  initialize = function(d_model, nhead, dim_feedforward = 2048, dropout = 0.1,
                        activation = "relu", layer_norm_eps = 1e-5,
                        batch_first = FALSE, norm_first = FALSE, bias = TRUE) {
    # Save flags
    self$norm_first <- norm_first
    
    # Multi-head self-attention module
    self$self_attn <- nn_multihead_attention(embed_dim = d_model, num_heads = nhead,
                                             dropout = dropout, bias = bias, batch_first = batch_first)
    # Feed-forward network components
    self$linear1 <- nn_linear(d_model, dim_feedforward, bias = bias)
    self$linear2 <- nn_linear(dim_feedforward, d_model, bias = bias)
    # Dropout layers
    self$dropout <- nn_dropout(dropout)    # used between linear1 and linear2
    self$dropout1 <- nn_dropout(dropout)   # used after self-attention
    self$dropout2 <- nn_dropout(dropout)   # used after feed-forward
    
    # LayerNorm layers (use elementwise_affine = bias; if bias=FALSE, no affine params)
    self$norm1 <- nn_layer_norm(normalized_shape = d_model, eps = layer_norm_eps, 
                                elementwise_affine = bias)
    self$norm2 <- nn_layer_norm(normalized_shape = d_model, eps = layer_norm_eps, 
                                elementwise_affine = bias)
    
    # Set up the activation function
    if (is.character(activation)) {
      act_name <- tolower(activation)
      if (act_name == "relu") {
        self$activation <- function(x) nnf_relu(x)
        self$activation_relu_or_gelu <- 1
      } else if (act_name == "gelu") {
        self$activation <- function(x) nnf_gelu(x)
        self$activation_relu_or_gelu <- 2
      } else {
        stop("Unsupported activation string: ", activation,
             ". Use 'relu', 'gelu', or a callable function.")
      }
    } else if (is.function(activation)) {
      self$activation <- activation
      # Identify if the function corresponds to ReLU or GELU for potential optimizations
      self$activation_relu_or_gelu <- 0
      # (In PyTorch, 1 indicates ReLU, 2 indicates GELU, 0 otherwise)
    } else {
      stop("activation must be a string ('relu' or 'gelu') or a function.")
    }
  },
  forward = function(src, src_mask = NULL, src_key_padding_mask = NULL, is_causal = FALSE) {
    # Validate mask usage with is_causal
    if (!is.null(src_mask) && is_causal) {
      stop("Explicit src_mask should not be set when is_causal=TRUE. Use one or the other.")
    }
    # If is_causal, generate a causal mask (upper triangular) for self-attention
    attn_mask_eff <- src_mask
    if (is_causal) {
      # Determine sequence length dimension based on batch_first
      if (self$self_attn$batch_first) {
        # src shape: (batch, seq, feature)
        L <- dim(src)[2]
      } else {
        # src shape: (seq, batch, feature)
        L <- dim(src)[1]
      }
      # Causal mask: True means position should not be attended (mask out future positions)
      attn_mask_eff <- torch_ones(c(L, L), dtype = torch_bool())
      attn_mask_eff <- attn_mask_eff$triu(diagonal = 1)  # ones above diagonal
    }
    
    # Self-attention with possible mask(s)
    # Query, Key, Value are all src (self-attention)
    if (self$norm_first) {
      # Pre-norm: normalize input before attention
      norm_src <- self$norm1(src)
      attn_out <- self$self_attn(norm_src, norm_src, norm_src, 
                                 attn_mask = attn_mask_eff, 
                                 key_padding_mask = src_key_padding_mask, 
                                 need_weights = FALSE)[[1]]
      # Add residual connection and apply dropout
      src2 <- src + self$dropout1(attn_out)
      # Feed-forward network with pre-norm
      norm_src2 <- self$norm2(src2)
      ff_out <- self$linear2(self$dropout(self$activation(self$linear1(norm_src2))))
      out <- src2 + self$dropout2(ff_out)
    } else {
      # Post-norm: attention on input directly
      attn_out <- self$self_attn(src, src, src, 
                                 attn_mask = attn_mask_eff, 
                                 key_padding_mask = src_key_padding_mask, 
                                 need_weights = FALSE)[[1]]
      # Residual connection then norm
      src2 <- src + self$dropout1(attn_out)
      out1 <- self$norm1(src2)
      # Feed-forward network on normalized output
      ff_out <- self$linear2(self$dropout(self$activation(self$linear1(out1))))
      # Residual connection then norm
      out2 <- out1 + self$dropout2(ff_out)
      out <- self$norm2(out2)
    }
    return(out)
  }
)


#' Transformer Encoder Module (R torch)
#'
#' Implements a stack of transformer encoder layers, optionally with a final layer normalization.
#'
#' @param encoder_layer (nn_module) an instance of `nn_transformer_encoder_layer` (or compatible) that defines the layer to be repeated.
#' @param num_layers (integer) the number of encoder layers to stack.
#' @param norm (nn_module or NULL) optional layer normalization module to apply after the last layer (e.g., `nn_layer_norm`). Default: NULL (no extra normalization).
#'
#' @details 
#' This module replicates the given `encoder_layer` `num_layers` times to construct the Transformer encoder. 
#' If a `norm` module is provided, it will be applied to the output of the final encoder layer. 
#' The forward pass sequentially applies each encoder layer to the input. 
#'
#' @returns An `nn_module` of class `nn_transformer_encoder`. Calling it on an input tensor of shape `(S, N, E)` or `(N, S, E)` (depending on `batch_first`) returns the encoded output of the same shape.
#'
#' @examples
#' if (torch_is_installed()) {
#'   layer <- nn_transformer_encoder_layer(d_model=32, nhead=4, batch_first=TRUE)
#'   model <- nn_transformer_encoder(layer, num_layers=2)
#'   x <- torch_randn(8, 5, 32)  # (batch, seq, feature) since batch_first=TRUE
#'   y <- model(x)  # output shape is (8, 5, 32)
#' }
#' @export
nn_transformer_encoder <- nn_module(
  "nn_transformer_encoder",
  initialize = function(encoder_layer, num_layers, norm = NULL) {
    # Replicate the encoder_layer for the specified number of layers
    if (!is_nn_module(encoder_layer)) {
      stop("encoder_layer must be an nn_module (transformer encoder layer instance).")
    }
    self$num_layers <- num_layers
    # Use clone_module to deep-copy the layer for each repetition
    self$layers <- nn_module_list(lapply(seq_len(num_layers), function(i) {
      clone_module(encoder_layer, deep = TRUE)
    }))
    self$norm <- norm
  },
  forward = function(src, mask = NULL, src_key_padding_mask = NULL, is_causal = FALSE) {
    output <- src
    # Pass through each encoder layer in sequence
    for (layer in self$layers) {
      output <- layer(output, src_mask = mask, src_key_padding_mask = src_key_padding_mask, 
                      is_causal = is_causal)
    }
    # Apply final normalization if provided
    if (!is.null(self$norm)) {
      output <- self$norm(output)
    }
    return(output)
  }
)
