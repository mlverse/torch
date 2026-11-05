#' Elu
#'
#' Applies element-wise,
#' \deqn{ELU(x) = max(0,x) + min(0, \alpha * (exp(x) - 1))}.
#'
#' @param input (N,*) tensor, where * means, any number of additional
#'   dimensions
#' @param alpha the alpha value for the ELU formulation. Default: 1.0
#' @param inplace can optionally do the operation in-place. Default: FALSE
#'
#' @examples
#' x <- torch_randn(2, 2)
#' y <- nnf_elu(x, alpha = 1)
#' nnf_elu_(x, alpha = 1)
#' torch_equal(x, y)
#' @export
nnf_elu <- function(input, alpha = 1, inplace = FALSE) {
  if (inplace) {
    torch_elu_(input, alpha = alpha)
  } else {
    torch_elu(input, alpha = alpha)
  }
}

#' @rdname nnf_elu
#' @export
nnf_elu_ <- function(input, alpha = 1) {
  torch_elu_(input, alpha = alpha)
}

#' Selu
#'
#' Applies element-wise,
#' \deqn{SELU(x) = scale * (max(0,x) + min(0, \alpha * (exp(x) - 1)))},
#' with \eqn{\alpha=1.6732632423543772848170429916717} and
#' \eqn{scale=1.0507009873554804934193349852946}.
#'
#' @inheritParams nnf_elu
#'
#' @examples
#' x <- torch_randn(2, 2)
#' y <- nnf_selu(x)
#' nnf_selu_(x)
#' torch_equal(x, y)
#' @export
nnf_selu <- function(input, inplace = FALSE) {
  if (inplace) {
    torch_selu_(input)
  } else {
    torch_selu(input)
  }
}

#' @rdname nnf_selu
#' @export
nnf_selu_ <- function(input) {
  nnf_selu(input, inplace = TRUE)
}

nnf_hardswish <- function(input, inplaxce = FALSE) {

}

#' Hardswish
#'
#' Applies the hardswish function, element-wise, as described in the paper:
#' Searching for MobileNetV3.
#'
#' \deqn{ \mbox{Hardswish}(x) = \left\{
#'   \begin{array}{ll}
#'   0 & \mbox{if } x \le -3, \\
#'   x & \mbox{if } x \ge +3, \\
#'   x \cdot (x + 3)/6 & \mbox{otherwise}
#'   \end{array}
#'   \right. }
#'
#' @inheritParams nnf_elu
#'
#' @export
nnf_hardswish <- function(input, inplace = FALSE) {
  not_implemented_error("not yet implemented.")
}

#' Hardshrink
#'
#' Applies the hard shrinkage function element-wise
#'
#' @inheritParams nnf_elu
#' @param lambd the lambda value for the Hardshrink formulation. Default: 0.5
#'
#' @export
nnf_hardshrink <- function(input, lambd = 0.5) {
  torch_hardshrink(input, lambd)
}

#' Hardtanh
#'
#' Applies the HardTanh function element-wise.
#'
#' @inheritParams nnf_elu
#' @param min_val minimum value of the linear region range. Default: -1
#' @param max_val maximum value of the linear region range. Default: 1
#'
#' @export
nnf_hardtanh <- function(input, min_val = -1, max_val = 1, inplace = FALSE) {
  if (inplace) {
    torch_hardtanh_(input, min_val, max_val)
  } else {
    torch_hardtanh(input, min_val, max_val)
  }
}

#' @rdname nnf_hardtanh
#' @export
nnf_hardtanh_ <- function(input, min_val = -1, max_val = 1) {
  nnf_hardtanh(input, min_val, max_val, inplace = TRUE)
}

#' Hardsigmoid
#'
#' Applies the element-wise function \eqn{\mbox{Hardsigmoid}(x) = \frac{ReLU6(x + 3)}{6}}
#'
#' @inheritParams nnf_elu
#' @param inplace NA If set to ``True``, will do this operation in-place. Default: ``False``
#'
#' @export
nnf_hardsigmoid <- function(input, inplace = FALSE) {
  if (inplace) {
    torch_hardsigmoid_(input)
  } else {
    torch_hardsigmoid(input)
  }
}


#' Leaky_relu
#'
#'
#' Applies element-wise,
#' \eqn{LeakyReLU(x) = max(0, x) + negative_slope * min(0, x)}
#'
#' @inheritParams nnf_elu
#' @param negative_slope Controls the angle of the negative slope. Default: 1e-2
#'
#' @export
nnf_leaky_relu <- function(input, negative_slope = 0.01, inplace = FALSE) {
  if (inplace) {
    torch_leaky_relu_(input, negative_slope)
  } else {
    torch_leaky_relu(input, negative_slope)
  }
}

#' Logsigmoid
#'
#' Applies element-wise \eqn{LogSigmoid(x_i) = log(\frac{1}{1 + exp(-x_i)})}
#'
#' @inheritParams nnf_elu
#'
#' @export
nnf_logsigmoid <- function(input) {
  torch_log_sigmoid(input)
}

#' Gumbel_softmax
#'
#' Samples from the Gumbel-Softmax distribution and
#' optionally discretizes.
#'
#' @param logits `[..., num_features]` unnormalized log probabilities
#' @param tau non-negative scalar temperature
#' @param hard if ``True``, the returned samples will be discretized as one-hot vectors,        but will be differentiated as if it is the soft sample in autograd
#' @param dim (int) A dimension along which softmax will be computed. Default: -1.
#'
#' @export
nnf_gumbel_softmax <- function(logits, tau = 1, hard = FALSE, dim = -1) {
  gumbels <- -torch_empty_like(logits, memory_format = torch_contiguous_format())
  gumbels <- gumbels$exponential_()$log()
  gumbels <- (logits + gumbels) / tau
  y_soft <- gumbels$softmax(dim)

  if (hard) {
    index <- y_soft$max(dim, keepdim = TRUE)[[2]]
    y_hard <- torch_zeros_like(logits, memory_format = torch_contiguous_format())
    y_hard <- y_hard$scatter_(dim, index, 1)
    ret <- y_hard - y_soft$detach() + y_soft
  } else {
    ret <- y_soft
  }
  ret
}

#' Softmax
#'
#' Applies a softmax function.
#'
#' Softmax is defined as:
#'
#' \deqn{Softmax(x_{i}) = exp(x_i)/\sum_j exp(x_j)}
#'
#' It is applied to all slices along dim, and will re-scale them so that the elements
#' lie in the range `[0, 1]` and sum to 1.
#'
#' @param input (Tensor) input
#' @param dim (int) A dimension along which softmax will be computed.
#' @param dtype (`torch.dtype`, optional) the desired data type of returned tensor.      If specified, the input tensor is casted to `dtype` before the operation      is performed. This is useful for preventing data type overflows.
#'   Default: NULL.
#'
#' @export
nnf_softmax <- function(input, dim, dtype = NULL) {
  if (is.null(dtype)) {
    ret <- input$softmax(dim)
  } else {
    ret <- input$softmax(dim, dtype = dtype)
  }

  ret
}

#' Softmin
#'
#' Applies a softmin function.
#'
#' Note that
#'
#' \deqn{Softmin(x) = Softmax(-x)}.
#'
#' See [nnf_softmax] definition for mathematical formula.
#'
#' @param input (Tensor) input
#' @param dim (int) A dimension along which softmin will be computed
#'   (so every slice        along dim will sum to 1).
#' @param dtype (`torch.dtype`, optional) the desired data type of returned tensor.      If specified, the input tensor is casted to `dtype` before the operation      is performed.
#'   This is useful for preventing data type overflows. Default: NULL.
#'
#' @export
nnf_softmin <- function(input, dim, dtype = NULL) {
  if (is.null(dtype)) {
    ret <- (-input)$softmax(dim)
  } else {
    ret <- (-input)$softmax(dim, dtype = dtype)
  }

  ret
}

#' Log_softmax
#'
#' Applies a softmax followed by a logarithm.
#'
#' While mathematically equivalent to log(softmax(x)), doing these two
#' operations separately is slower, and numerically unstable. This function
#' uses an alternative formulation to compute the output and gradient correctly.
#'
#' @param input (Tensor) input
#' @param dim (int) A dimension along which log_softmax will be computed.
#' @param dtype (`torch.dtype`, optional) the desired data type of returned tensor.
#'   If specified, the input tensor is casted to `dtype` before the operation
#'   is performed. This is useful for preventing data type overflows.
#'   Default: `NULL`.
#'
#' @export
nnf_log_softmax <- function(input, dim = NULL, dtype = NULL) {
  if (is.null(dtype)) {
    ret <- input$log_softmax(dim)
  } else {
    ret <- input$log_softmax(dim, dtype = dtype)
  }

  ret
}

#' Glu
#'
#' The gated linear unit. Computes:
#'
#' \deqn{GLU(a, b) = a \otimes \sigma(b)}
#'
#' where `input` is split in half along `dim` to form `a` and `b`, \eqn{\sigma}
#' is the sigmoid function and \eqn{\otimes} is the element-wise product
#' between matrices.
#'
#' See [Language Modeling with Gated Convolutional Networks](https://arxiv.org/abs/1612.08083).
#'
#' @param input (Tensor) input tensor
#' @param dim (int) dimension on which to split the input. Default: -1
#'
#' @export
nnf_glu <- function(input, dim = -1) {
  torch_glu(self = input, dim = dim)
}

#' Gelu
#'
#' @section gelu(input) -> Tensor :
#'
#' Applies element-wise the function
#' \eqn{GELU(x) = x * \Phi(x)}
#'
#' where \eqn{\Phi(x)} is the Cumulative Distribution Function for
#' Gaussian Distribution.
#'
#' See \href{https://arxiv.org/abs/1606.08415}{Gaussian Error Linear Units (GELUs)}.
#'
#' @inheritParams nnf_elu
#' @param approximate By default it's none, and applies element-wise x*pnorm(x),
#'   if 'tanh', then GELU is estimated. See [GELU](https://arxiv.org/abs/1606.08415) for
#'   more info.
#'
#' @export
nnf_gelu <- function(input, approximate = "none") {
  torch_gelu(self = input, approximate = approximate)
}

#' Prelu
#'
#' Applies element-wise the function
#' \eqn{PReLU(x) = max(0,x) + weight * min(0,x)}
#' where weight is a learnable parameter.
#'
#' @inheritParams nnf_elu
#' @param weight (Tensor) the learnable weights
#'
#' @export
nnf_prelu <- function(input, weight) {
  torch_prelu(input, weight)
}

#' Relu
#'
#' Applies the rectified linear unit function element-wise.
#'
#' @inheritParams nnf_elu
#'
#' @export
nnf_relu <- function(input, inplace = FALSE) {
  if (inplace) {
    torch_relu_(input)
  } else {
    torch_relu(input)
  }
}

#' Relu6
#'
#' Applies the element-wise function \eqn{ReLU6(x) = min(max(0,x), 6)}.
#' @inheritParams nnf_elu
#'
#' @export
nnf_relu6 <- function(input, inplace = FALSE) {
  nnf_hardtanh(input, 0, 6, inplace)
}

#' @rdname nnf_relu
#' @export
nnf_relu_ <- function(input) {
  nnf_relu(input, inplace = TRUE)
}

#' Rrelu
#'
#' Randomized leaky ReLU.
#'
#' @inheritParams nnf_elu
#' @param lower lower bound of the uniform distribution. Default: 1/8
#' @param upper upper bound of the uniform distribution. Default: 1/3
#' @param training bool wether it's a training pass. DEfault: FALSE
#'
#' @export
nnf_rrelu <- function(input, lower = 1 / 8, upper = 1 / 3, training = FALSE,
                      inplace = FALSE) {
  if (inplace) {
    result <- torch_rrelu_(input, lower, upper, training)
  } else {
    result <- torch_rrelu(input, lower, upper, training)
  }

  result
}

#' @rdname nnf_rrelu
#' @export
nnf_rrelu_ <- function(input, lower = 1 / 8, upper = 1 / 3, training = FALSE) {
  nnf_rrelu(input, lower, upper, training = TRUE)
}

#' Celu
#'
#' Applies element-wise, \eqn{CELU(x) = max(0,x) + min(0, \alpha * (exp(x \alpha) - 1))}.
#'
#' @inheritParams nnf_elu
#' @param alpha the alpha value for the CELU formulation. Default: 1.0
#'
#' @export
nnf_celu <- function(input, alpha = 1, inplace = FALSE) {
  if (inplace) {
    torch_celu_(input, alpha)
  } else {
    torch_celu(input, alpha)
  }
}

#' @rdname nnf_celu
#' @export
nnf_celu_ <- function(input, alpha = 1) {
  torch_celu_(input, alpha)
}

#' Softplus
#'
#' Applies element-wise, the function \eqn{Softplus(x) = 1/\beta * log(1 + exp(\beta * x))}.
#'
#' For numerical stability the implementation reverts to the linear function
#' when \eqn{input * \beta > threshold}.
#'
#' @inheritParams nnf_elu
#' @param beta the beta value for the Softplus formulation. Default: 1
#' @param threshold values above this revert to a linear function. Default: 20
#'
#' @export
nnf_softplus <- function(input, beta = 1, threshold = 20) {
  torch_softplus(input, beta, threshold)
}

#' Softshrink
#'
#' Applies the soft shrinkage function elementwise
#'
#' @inheritParams nnf_elu
#' @param lambd the lambda (must be no less than zero) value for the Softshrink
#'   formulation. Default: 0.5
#'
#' @export
nnf_softshrink <- function(input, lambd = 0.5) {
  torch_softshrink(input, lambd)
}

#' Softsign
#'
#' Applies element-wise, the function \eqn{SoftSign(x) = x/(1 + |x|}
#'
#' @inheritParams nnf_elu
#'
#' @export
nnf_softsign <- function(input) {
  input / (input$abs() + 1)
}

#' Tanhshrink
#'
#' Applies element-wise, \eqn{Tanhshrink(x) = x - Tanh(x)}
#'
#' @inheritParams nnf_elu
#'
#' @export
nnf_tanhshrink <- function(input) {
  input - input$tanh()
}

#' Threshold
#'
#' Thresholds each element of the input Tensor.
#' @inheritParams nnf_elu
#' @param threshold The value to threshold at
#' @param value The value to replace with
#'
#' @export
nnf_threshold <- function(input, threshold, value, inplace = FALSE) {
  if (inplace) {
    torch_threshold_(input, threshold, value)
  } else {
    torch_threshold(input, threshold, value)
  }
}

#' @rdname nnf_threshold
#' @export
nnf_threshold_ <- function(input, threshold, value) {
  nnf_threshold(input, threshold, value, TRUE)
}

#' Multi head attention forward
#'
#' Allows the model to jointly attend to information from different representation
#' subspaces. See reference: Attention Is All You Need
#'
#' @param embed_dim_to_check  total dimension of the model.
#' @param num_heads  parallel attention heads.
#' @param in_proj_weight  input projection weight.
#' @param bias_k bias of the key and value sequences to be added at dim=0.
#' @param add_zero_attn  add a new batch of zeros to the key and
#'   value sequences at dim=1.
#' @param dropout_p  probability of an element to be zeroed.
#' @param out_proj_weight the output projection weight.
#' @param training  apply dropout if is `TRUE`.
#' @param key_padding_mask  if provided, specified padding elements in the key will
#'   be ignored by the attention. This is an binary mask. When the value is True
#'   the corresponding value on the attention layer will be filled with -inf.
#' @param need_weights  output attn_output_weights.
#' @param attn_mask  2D or 3D mask that prevents attention to certain positions.
#'   This is an additive mask (i.e. the values will be added to the attention layer).
#'   A 2D mask will be broadcasted for all the batches while a 3D mask allows to
#'   specify a different mask for the entries of each batch.
#' @param use_separate_proj_weight  the function accept the proj. weights for
#'   query, key, and value in different forms. If false, in_proj_weight will be used,
#'   which is a combination of q_proj_weight, k_proj_weight, v_proj_weight.
#' @param q_proj_weight input projection weight and bias.
#' @param static_k static key and value used for attention operators.
#' @param query \eqn{(L, N, E)} where L is the target sequence length, N is the batch size, E is
#'   the embedding dimension. If batch_first is TRUE, the first two dimensions are transposed.
#' @param key \eqn{(S, N, E)}, where S is the source sequence length, N is the batch size, E is
#'   the embedding dimension. If batch_first is TRUE, the first two dimensions are transposed.
#' @param value \eqn{(S, N, E)} where S is the source sequence length, N is the batch size, E is
#'   the embedding dimension. If batch_first is TRUE, the first two dimensions are transposed.
#' @param key_padding_mask \eqn{(N, S)} where N is the batch size, S is the source sequence length.
#'   If a ByteTensor is provided, the non-zero positions will be ignored while the position
#'   with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
#'   value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
#' @param attn_mask 2D mask \eqn{(L, S)} where L is the target sequence length, S is the source sequence length.
#'   3D mask \eqn{(N*num_heads, L, S)} where N is the batch size, L is the target sequence length,
#'   S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
#'   positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
#'   while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
#'   is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
#'   is provided, it will be added to the attention weight.
#' @param avg_weights Logical; whether to average attn_output_weights over the
#'   attention heads before outputting them. This doesn't change the returned
#'   value of attn_output; it only affects the returned attention weight matrix.
#' @param in_proj_bias input projection bias.
#' @param bias_v currently undocumented.
#' @param out_proj_bias output projection bias.
#' @param k_proj_weight currently undocumented.
#' @param v_proj_weight currently undocumented.
#' @param static_v currently undocumented.
#' @param batch_first Logical; whether to expect query, key, and value to have 
#'   batch as their first parameter, and to return output with batch first.
#' @param is_causal Logical; a hint that `attn_mask` is the causal mask, which then has to be provided.
#'   If `need_weights` is `FALSE` and there is no `key_padding_mask`, the causal mask is applied by
#'   [torch_scaled_dot_product_attention()] instead of `attn_mask`, which allows faster kernels.
#'
#' @details
#' When `need_weights` is `FALSE`, the attention is computed with the fused
#' [torch_scaled_dot_product_attention()], like in PyTorch.
#'
#' @export
nnf_multi_head_attention_forward <- function(query, # type: Tensor
                                             key, # type: Tensor
                                             value, # type: Tensor
                                             embed_dim_to_check, # type: int
                                             num_heads, # type: int
                                             in_proj_weight, # type: Tensor
                                             in_proj_bias, # type: Tensor
                                             bias_k, # type: Optional[Tensor]
                                             bias_v, # type: Optional[Tensor]
                                             add_zero_attn, # type: bool
                                             dropout_p, # type: float
                                             out_proj_weight, # type: Tensor
                                             out_proj_bias, # type: Tensor
                                             training = TRUE, # type: bool
                                             key_padding_mask = NULL, # type: Optional[Tensor]
                                             need_weights = TRUE, # type: bool
                                             attn_mask = NULL, # type: Optional[Tensor]
                                             avg_weights = TRUE, # type: bool
                                             use_separate_proj_weight = FALSE, # type: bool
                                             q_proj_weight = NULL, # type: Optional[Tensor]
                                             k_proj_weight = NULL, # type: Optional[Tensor]
                                             v_proj_weight = NULL, # type: Optional[Tensor]
                                             static_k = NULL, # type: Optional[Tensor]
                                             static_v = NULL, # type: Optional[Tensor])
                                             batch_first = FALSE, # type: bool
                                             is_causal = FALSE # type: bool
) {
  # Like in PyTorch, self-attention is detected by checking whether query, key and value are the same tensor
  # (and not by comparing their values, which is expensive). This has to happen before transposing them.
  key_is_value <- is_same_tensor(key, value)
  self_attention <- key_is_value && is_same_tensor(query, key)

  # Without the attention weights, we use the fused scaled dot product attention, like PyTorch does. In the
  # common case where nothing is added to the keys and values, q, k and v only need to be viewed in its layout
  # (bsz, num_heads, length, head_dim), so inputs with the batch first don't have to be transposed.
  fast_path <- !need_weights && is.null(bias_k) && is.null(bias_v) && !add_zero_attn && is.null(static_k) &&
    is.null(static_v)
  if (batch_first && !fast_path) {
    query <- torch_transpose(query, 1, 2)
    key <- torch_transpose(key, 1, 2)
    value <- torch_transpose(value, 1, 2)
  }
  # whether query, key and value are of shape (bsz, length, embed_dim) instead of (length, bsz, embed_dim)
  is_bsz_first <- batch_first && fast_path
  o <- query$size()
  if (is_bsz_first) {
    bsz <- o[[1]]
    tgt_len <- o[[2]]
    src_len <- key$size(2)
  } else {
    tgt_len <- o[[1]]
    bsz <- o[[2]]
    src_len <- key$size(1)
  }
  embed_dim <- o[[3]]

  head_dim <- embed_dim %/% num_heads
  if (head_dim * num_heads != embed_dim) {
    value_error("embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
  }

  attn_mask <- mha_canonical_mask(attn_mask, "attn_mask", query$dtype)
  key_padding_mask <- mha_canonical_mask(key_padding_mask, "key_padding_mask", query$dtype)
  if (is_causal && is.null(attn_mask)) {
    runtime_error(paste0(
      "Need attn_mask if specifying the is_causal hint. You may use ",
      "`nn_transformer_encoder_layer()` with `is_causal = TRUE` to create this mask."
    ))
  }
  if (!is.null(attn_mask)) {
    mha_check_attn_mask(attn_mask, bsz, num_heads, tgt_len, src_len)
  }
  # With the is_causal hint, the fused attention can apply the causal mask itself, which allows using the
  # fastest kernels. A key padding mask is merged into the attention mask, which is then no longer causal.
  if (is_causal && fast_path && is.null(key_padding_mask)) {
    attn_mask <- NULL
  } else {
    is_causal <- FALSE
  }

  if (!use_separate_proj_weight) {
    if (self_attention) {
      o <- nnf_linear(query, in_proj_weight, in_proj_bias)$chunk(3, dim = -1)
      q <- o[[1]]
      k <- o[[2]]
      v <- o[[3]]
    } else {
      w <- in_proj_weight$split(c(embed_dim, 2 * embed_dim))
      b <- if (is.null(in_proj_bias)) list(NULL, NULL) else in_proj_bias$split(c(embed_dim, 2 * embed_dim))
      q <- nnf_linear(query, w[[1]], b[[1]])
      if (key_is_value) {
        # encoder-decoder attention
        o <- nnf_linear(key, w[[2]], b[[2]])$chunk(2, dim = -1)
        k <- o[[1]]
        v <- o[[2]]
      } else {
        w <- w[[2]]$chunk(2)
        b <- if (is.null(in_proj_bias)) list(NULL, NULL) else b[[2]]$chunk(2)
        k <- nnf_linear(key, w[[1]], b[[1]])
        v <- nnf_linear(value, w[[2]], b[[2]])
      }
    }
  } else {
    b <- if (is.null(in_proj_bias)) list(NULL, NULL, NULL) else in_proj_bias$chunk(3)
    q <- nnf_linear(query, q_proj_weight, b[[1]])
    k <- nnf_linear(key, k_proj_weight, b[[2]])
    v <- nnf_linear(value, v_proj_weight, b[[3]])
  }

  if (fast_path) {
    to_heads <- if (is_bsz_first) {
      function(x) x$view(c(bsz, -1, num_heads, head_dim))$transpose(2, 3)
    } else {
      function(x) x$view(c(-1, bsz, num_heads, head_dim))$permute(c(2, 3, 1, 4))
    }
    if (!is.null(key_padding_mask)) {
      mha_check_key_padding_mask(key_padding_mask, bsz, src_len)
    }
    return(mha_sdpa(
      to_heads(q), to_heads(k), to_heads(v), attn_mask, key_padding_mask, is_causal, dropout_p, training,
      out_proj_weight, out_proj_bias, batch_first
    ))
  }

  if (!is.null(bias_k) && !is.null(bias_v)) {
    if (is.null(static_k) && is.null(static_v)) {
      k <- torch_cat(list(k, bias_k$`repeat`(c(1, bsz, 1))))
      v <- torch_cat(list(v, bias_v$`repeat`(c(1, bsz, 1))))
      if (!is.null(attn_mask)) {
        attn_mask <- nnf_pad(attn_mask, c(0, 1))
      }
      if (!is.null(key_padding_mask)) {
        key_padding_mask <- nnf_pad(key_padding_mask, c(0, 1))
      }
    } else {
      value_error("bias cannot be added to static key and value.")
    }
  }

  # (length, bsz, embed_dim) -> (bsz * num_heads, length, head_dim)
  q <- q$reshape(c(tgt_len, bsz * num_heads, head_dim))$transpose(1, 2)
  k <- if (is.null(static_k)) k$reshape(c(-1, bsz * num_heads, head_dim))$transpose(1, 2) else static_k
  v <- if (is.null(static_v)) v$reshape(c(-1, bsz * num_heads, head_dim))$transpose(1, 2) else static_v

  src_len <- k$size(2)
  if (!is.null(key_padding_mask)) {
    mha_check_key_padding_mask(key_padding_mask, bsz, src_len)
  }

  if (add_zero_attn) {
    src_len <- src_len + 1
    k <- torch_cat(list(k, k$new_zeros(c(bsz * num_heads, 1, head_dim))), dim = 2)
    v <- torch_cat(list(v, v$new_zeros(c(bsz * num_heads, 1, head_dim))), dim = 2)
    if (!is.null(attn_mask)) {
      attn_mask <- nnf_pad(attn_mask, c(0, 1))
    }
    if (!is.null(key_padding_mask)) {
      key_padding_mask <- nnf_pad(key_padding_mask, c(0, 1))
    }
  }

  if (!need_weights) {
    # splitting the first dimension of (bsz * num_heads, length, head_dim) is free
    return(mha_sdpa(
      q$view(c(bsz, num_heads, tgt_len, head_dim)),
      k$reshape(c(bsz, num_heads, src_len, head_dim)),
      v$reshape(c(bsz, num_heads, src_len, head_dim)),
      attn_mask, key_padding_mask, is_causal, dropout_p, training, out_proj_weight, out_proj_bias, batch_first
    ))
  }

  attn_output_weights <- torch_bmm(q * head_dim^(-0.5), k$transpose(2, 3))
  if (!is.null(attn_mask)) {
    attn_output_weights <- attn_output_weights + attn_mask
  }
  if (!is.null(key_padding_mask)) {
    attn_output_weights <- attn_output_weights$view(c(bsz, num_heads, tgt_len, src_len)) +
      key_padding_mask$reshape(c(bsz, 1, 1, src_len))
    attn_output_weights <- attn_output_weights$view(c(bsz * num_heads, tgt_len, src_len))
  }

  attn_output_weights <- nnf_softmax(attn_output_weights, dim = -1)
  attn_output_weights <- nnf_dropout(attn_output_weights, p = dropout_p, training = training)

  attn_output <- torch_bmm(attn_output_weights, v)
  attn_output <- attn_output$transpose(1, 2)$reshape(c(tgt_len, bsz, embed_dim))
  attn_output <- nnf_linear(attn_output, out_proj_weight, out_proj_bias)
  if (batch_first) {
    attn_output <- torch_transpose(attn_output, 1, 2)
  }

  attn_output_weights <- attn_output_weights$view(c(bsz, num_heads, tgt_len, src_len))
  if (avg_weights) {
    attn_output_weights <- attn_output_weights$mean(dim = 2)
  }
  list(attn_output, attn_output_weights)
}

# Whether `x` and `y` are the same tensor object.
is_same_tensor <- function(x, y) {
  identical(x, y) || (inherits(x, "torch_tensor") && inherits(y, "torch_tensor") &&
    identical(xptr_address(x), xptr_address(y)))
}

# Converts a mask of `nnf_multi_head_attention_forward()` into an additive mask of type `dtype`, where boolean
# (and, as documented, integer) masks are `-Inf` where they are `TRUE` (non-zero).
mha_canonical_mask <- function(mask, name, dtype) {
  if (is.null(mask)) {
    return(NULL)
  }
  if (mask$is_floating_point()) {
    return(mask$to(dtype = dtype))
  }
  if (mask$dtype != torch_bool()) {
    if (mask$is_complex()) {
      value_error("Only boolean, integer and floating point types of {name} are supported.")
    }
    mask <- mask$to(dtype = torch_bool())
  }
  torch_zeros_like(mask, dtype = dtype)$masked_fill_(mask, -Inf)
}

mha_check_attn_mask <- function(attn_mask, bsz, num_heads, tgt_len, src_len) {
  expected <- switch(as.character(attn_mask$dim()),
    "2" = c(tgt_len, src_len),
    "3" = c(bsz * num_heads, tgt_len, src_len),
    value_error("attn_mask's dimension {attn_mask$dim()} is not supported.")
  )
  if (!identical(as.numeric(attn_mask$shape), as.numeric(expected))) {
    value_error(paste0(
      "The shape of the {attn_mask$dim()}D attn_mask is ({paste(attn_mask$shape, collapse = ', ')}), ",
      "but should be ({paste(expected, collapse = ', ')})."
    ))
  }
}

mha_check_key_padding_mask <- function(key_padding_mask, bsz, src_len) {
  if (!identical(as.numeric(key_padding_mask$shape), as.numeric(c(bsz, src_len)))) {
    value_error(paste0(
      "Expected key_padding_mask of shape ({bsz}, {src_len}), ",
      "but got ({paste(key_padding_mask$shape, collapse = ', ')})."
    ))
  }
}

# Computes the attention output of `nnf_multi_head_attention_forward()` with the fused scaled dot product
# attention, for `q`, `k` and `v` of shape (bsz, num_heads, length, head_dim) and the additive masks.
mha_sdpa <- function(q, k, v, attn_mask, key_padding_mask, is_causal, dropout_p, training, out_proj_weight,
                     out_proj_bias, batch_first) {
  bsz <- q$size(1)
  tgt_len <- q$size(3)
  src_len <- k$size(3)
  embed_dim <- q$size(2) * q$size(4)
  # The attention mask (tgt_len, src_len) broadcasts, (bsz * num_heads, tgt_len, src_len) is split like q, k and
  # v, and the key padding mask is merged into it, as the fused attention has no separate one.
  mask <- attn_mask
  if (!is.null(mask) && mask$dim() == 3) {
    mask <- mask$reshape(c(bsz, -1, tgt_len, src_len))
  }
  if (!is.null(key_padding_mask)) {
    key_padding_mask <- key_padding_mask$reshape(c(bsz, 1, 1, src_len))
    mask <- if (is.null(mask)) key_padding_mask else mask + key_padding_mask
  }
  attn_output <- torch_scaled_dot_product_attention(
    q, k, v,
    attn_mask = mask %||% list(),
    dropout_p = if (training) dropout_p else 0,
    is_causal = is_causal
  )
  # (bsz, num_heads, tgt_len, head_dim) -> (bsz, tgt_len, embed_dim) or (tgt_len, bsz, embed_dim)
  attn_output <- if (batch_first) {
    attn_output$transpose(2, 3)$reshape(c(bsz, tgt_len, embed_dim))
  } else {
    attn_output$permute(c(3, 1, 2, 4))$reshape(c(tgt_len, bsz, embed_dim))
  }
  list(nnf_linear(attn_output, out_proj_weight, out_proj_bias), NULL)
}

#' Sigmoid
#'
#' Applies element-wise \eqn{Sigmoid(x_i) = \frac{1}{1 + exp(-x_i)}}
#'
#' @inheritParams nnf_elu
#'
#' @export
nnf_sigmoid <- function(input) {
  torch_sigmoid(input)
}

#' Sparsemax
#'
#' Applies the SparseMax activation.
#'
#' @details
#' The SparseMax activation is described in
#' ['From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification'](https://arxiv.org/abs/1602.02068)
#' The implementation is based on [aced125/sparsemax](https://github.com/aced125/sparsemax/tree/master/sparsemax)
#'
#' @param input the input tensor
#' @param dim The dimension over which to apply the sparsemax function. (-1)
#'
#' @export
nnf_contrib_sparsemax <- function(input, dim = -1) {
  if (!is_torch_tensor(input)) {
    value_error("Input should be a tensor and got '{class(input)}.")
  }

  dim <- as_1_based_dim(dim)

  ptr <- cpp_contrib_torch_sparsemax(input$ptr, dim)

  Tensor$new(ptr = ptr)
}

#' Applies the Sigmoid Linear Unit (SiLU) function, element-wise.
#' See [nn_silu()] for more information.
#' @seealso [nn_silu()].
#' @inheritParams nnf_relu
#' @export
nnf_silu <- function(input, inplace = FALSE) {
  if (inplace) {
    torch_silu_(input)
  } else {
    torch_silu(input)
  }
}
