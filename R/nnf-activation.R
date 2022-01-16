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
#'
#' @export
nnf_gelu <- function(input) {
  torch_gelu(self = input)
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
#' @param query map a query and a set of key-value pairs to an output.
#'   See "Attention Is All You Need" for more details.
#' @param embed_dim_to_check  total dimension of the model.
#' @param num_heads  parallel attention heads.
#' @param in_proj_weight  input projection weight and bias.
#' @param bias_k bias of the key and value sequences to be added at dim=0.
#' @param add_zero_attn  add a new batch of zeros to the key and
#'   value sequences at dim=1.
#' @param dropout_p  probability of an element to be zeroed.
#' @param out_proj_weight the output projection weight and bias.
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
#'   the embedding dimension.
#' @param key \eqn{(S, N, E)}, where S is the source sequence length, N is the batch size, E is
#'   the embedding dimension.
#' @param value \eqn{(S, N, E)} where S is the source sequence length, N is the batch size, E is
#'   the embedding dimension.
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
#' @param in_proj_bias currently undocumented.
#' @param bias_v currently undocumented.
#' @param out_proj_bias currently undocumented.
#' @param k_proj_weight currently undocumented.
#' @param v_proj_weight currently undocumented.
#' @param static_v currently undocumented.
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
                                             static_v = NULL # type: Optional[Tensor])
) {
  o <- query$size()
  tgt_len <- o[[1]]
  bsz <- o[[2]]
  embed_dim <- o[[3]]


  head_dim <- floor(embed_dim / num_heads)

  scaling <- head_dim^(-0.5)

  if (!use_separate_proj_weight) {
    if (torch_equal(query, key) && torch_equal(key, value)) {
      # self-attention
      o <- nnf_linear(query, in_proj_weight, in_proj_bias)$chunk(3, dim = -1)
      q <- o[[1]]
      k <- o[[2]]
      v <- o[[3]]
    } else if (torch_equal(key, value)) {
      # encoder-decoder attention
      #             # This is inline in_proj function with in_proj_weight and in_proj_bias
      b_ <- in_proj_bias
      start_ <- 1
      end_ <- embed_dim
      w_ <- in_proj_weight[start_:end_, ]

      if (!is.null(b_)) {
        b_ <- b_[start_:end_]
      }

      q <- nnf_linear(query, w_, b_)

      if (is.null(key)) {
        k <- NULL
        v <- NULL
      } else {
        b_ <- in_proj_bias
        start_ <- embed_dim + 1
        end_ <- NULL
        w_ <- in_proj_weight[start_:N, ]
        if (!is.null(b_)) {
          b_ <- b_[start_:N]
          o <- nnf_linear(key, w_, b_)$chunk(2, dim = -1)
          k <- o[[1]]
          v <- o[[2]]
        }
      }
    } else {

      # This is inline in_proj function with in_proj_weight and in_proj_bias
      b_ <- in_proj_bias
      start_ <- 1
      end_ <- embed_dim
      w_ <- in_proj_weight[start_:end_, ]
      if (!is.null(b_)) {
        b_ <- b_[start_:end_]
      }
      q <- nnf_linear(query, w_, b_)


      # This is inline in_proj function with in_proj_weight and in_proj_bias
      b_ <- in_proj_bias
      start_ <- embed_dim + 1
      end_ <- embed_dim * 2
      w_ <- in_proj_weight[start_:end_, ]
      if (!is.null(b_)) {
        b_ <- b_[start_:end_]
      }
      k <- nnf_linear(key, w_, b_)

      # This is inline in_proj function with in_proj_weight and in_proj_bias
      b_ <- in_proj_bias
      start_ <- embed_dim * 2 + 1
      end_ <- NULL
      w_ <- in_proj_weight[start_:N, ]
      if (!is.null(b_)) {
        b_ <- b_[start_:N]
      }
      v <- nnf_linear(value, w_, b_)
    }
  } else {
    if (!is.null(in_proj_bias)) {
      q <- nnf_linear(query, q_proj_weight, in_proj_bias[1:embed_dim])
      k <- nnf_linear(key, k_proj_weight, in_proj_bias[embed_dim:(embed_dim * 2)])
      v <- nnf_linear(value, v_proj_weight, in_proj_bias[(embed_dim * 2):N])
    } else {
      q <- nnf_linear(query, q_proj_weight, in_proj_bias)
      k <- nnf_linear(key, k_proj_weight, in_proj_bias)
      v <- nnf_linear(value, v_proj_weight, in_proj_bias)
    }
  }

  q <- q * scaling

  if (!is.null(bias_k) && !is.null(bias_v)) {
    if (is.null(static_k) && is.null(static_v)) {
      k <- torch_cat(list(k, bias_k[["repeat"]](c(1, bsz, 1))))
      v <- torch_cat(list(v, bias_v[["repeat"]](c(1, bsz, 1))))

      if (!is.null(attn_mask)) {
        attn_mask <- nnf_pad(attn_mask, c(0, 1))
      }

      if (!is.null(key_padding_mask)) {
        key_padding_mask <- nnf_pad(key_padding_mask, c(0, 1))
      }
    }
  }

  q <- q$contiguous()$view(c(tgt_len, bsz * num_heads, head_dim))$transpose(1, 2)

  if (!is.null(k)) {
    k <- k$contiguous()$view(c(-1, bsz * num_heads, head_dim))$transpose(1, 2)
  }

  if (!is.null(v)) {
    v <- v$contiguous()$view(c(-1, bsz * num_heads, head_dim))$transpose(1, 2)
  }


  if (!is.null(static_k)) {
    k <- static_k
  }

  if (!is.null(static_v)) {
    v <- static_v
  }

  src_len <- k$size(2)

  if (add_zero_attn) {
    src_len <- src_len + 1
    k_size <- k$size()
    k <- torch_cat(list(k, torch_zeros(append(list(k_size[1], 1), k_size[3:length(k_size)]),
      dtype = k$dtype, device = k$device
    )), dim = 2)
    v_size <- v$size()
    k <- torch_cat(list(k, torch_zeros(append(list(v_size[1], 1), v_size[3:length(v_size)]),
      dtype = v$dtype, device = v$device
    )), dim = 2)

    if (!is.null(attn_mask)) {
      attn_mask <- nnf_pad(attn_mask, list(0, 1))
    }

    if (!is.null(key_padding_mask)) {
      key_padding_mask <- nnf_pad(key_padding_mask, list(0, 1))
    }
  }

  attn_output_weights <- torch_bmm(q, k$transpose(2, 3))

  if (!is.null(attn_mask)) {
    if (attn_mask$dtype == torch_bool()) {
      attn_output_weights$masked_fill_(attn_mask, -Inf)
    } else {
      attn_output_weights <- attn_output_weights + attn_mask
    }
  }

  if (!is.null(key_padding_mask)) {
    attn_output_weights <- attn_output_weights$view(c(bsz, num_heads, tgt_len, src_len))
    attn_output_weights <- attn_output_weights$masked_fill(
      key_padding_mask$unsqueeze(2)$unsqueeze(3),
      -Inf
    )
    attn_output_weights <- attn_output_weights$view(c(
      bsz * num_heads,
      tgt_len,
      src_len
    ))
  }

  attn_output_weights <- nnf_softmax(attn_output_weights, dim = -1)
  attn_output_weights <- nnf_dropout(attn_output_weights,
    p = dropout_p,
    training = training
  )

  attn_output <- torch_bmm(attn_output_weights, v)
  attn_output <- attn_output$transpose(1, 2)$contiguous()$view(c(tgt_len, bsz, embed_dim))
  attn_output <- nnf_linear(attn_output, out_proj_weight, out_proj_bias)

  if (need_weights) {
    attn_output_weights <- attn_output_weights$view(c(
      bsz, num_heads,
      tgt_len,
      src_len
    ))
    if (avg_weights) {
      return(list(attn_output, attn_output_weights$sum(dim = 2) / num_heads))
    } else {
      return(list(attn_output, attn_output_weights))
    }
  } else {
    return(list(attn_output, NULL))
  }
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
