#' Elu
#'
#' Applies element-wise,
#' \deqn{ELU(x) = max(0,x) + min(0, \alpha * (exp(x) - 1))}.
#'
#' @param input (N,∗) tensor, where * means, any number of additional 
#'   dimensions
#' @param alpha the alpha value for the ELU formulation. Default: 1.0
#' @param inplace can optionally do the operation in-place. Default: FALSE
#'
#' @export
nnf_elu <- function(input, alpha=1, inplace=FALSE) {
  if(inplace)
    torch_elu_(input, alpha = alpha)
  else
    torch_elu(input, alpha = alpha)
}

#' @rdname nnf_elu
#' @export
nnf_elu_ <- function(input, alpha=1) {
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
#' @export
nnf_selu <- function(input, inplace = FALSE) {
  if (inplace)
    torch_selu_(input)
  else
    torch_selu(input)
}

#' @rdname nnf_selu
#' @export
nnf_selu_ <- function(input) {
  nnf_selu(input, inplace = TRUE)
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
  if (inplace)
    torch_hardtanh_(input, min_val, max_val)
  else
    torch_hardtanh(input, min_val, max_val)
}

#' @rdname nnf_hardtanh
#' @export
nnf_hardtanh_ <- function(input, min_val = -1, max_val = 1) {
  nnf_hardtanh(input, min_val, max_val, inplace = TRUE)
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
  if (inplace)
    torch_leaky_relu_(input, negative_slope)
  else
    torch_leaky_relu(input, negative_slope)
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
    ret = y_soft
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
  if (is.null(dtype))
    ret <- input$softmax(dim)
  else
    ret <- input$softmax(dim, dtype = dtype)
  
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
  if (is.null(dtype))
    ret <- (-input)$softmax(dim)
  else
    ret <- (-input)$softmax(dim, dtype = dtype)
  
  ret
}

nnf_log_softmax <- function(input, dim = NULL, dtype = NULL, ...) {
  
  if (is.null(dtype))
    ret <- input$log_softmax(dim())
  else
    ret <- input$log_softmax(dim, dtype = dtype)
  
  ret  
}

nnf_glu <- function(input, dim = -1) {
  torch_glu(self = input, dim = dim)
}

nnf_gelu <- function(input) {
  torch_gelu(self = input)
}

nnf_prelu <- function(input, weight) {
  torch_prelu(input, weight)
}

nnf_relu <- function(input, inplace = FALSE) {
  if (inplace)
    torch_relu_(input)
  else
    torch_relu(input)
}

nnf_relu6 <- function(input, inplace = FALSE) {
  nnf_hardtanh(input, 0, 6, inplace)
}

nnf_relu_ <- function(input) {
  nnf_relu(input, inplace = TRUE)
}

nnf_rrelu <- function(input, lower = 1/8, upper = 1/3, training = FALSE,
                      inplace = FALSE) {
  
  if (inplace)
    result <- torch_rrelu_(input, lower, upper, training)
  else
    result <- torch_rrelu(input, lower, upper, training)
  
  result
}

nnf_rrelu_ <- function(input, lower = 1/8, upper = 1/3, training = FALSE) {
  nnf_rrelu(input, lower, upper, training = TRUE)
}

nnf_celu <- function(input, alpha = 1, inplace = FALSE) {
  if (inplace)
    torch_celu_(input, alpha)
  else
    torch_celu(input, alpha)
}

nnf_celu_ <- function(input, alpha = 1) {
  torch_celu_(input, alpha)
}

nnf_softplus <- function(input, beta = 1, threshold = 20) {
  torch_softplus(input, beta, threshold)
}

nnf_softshrink <- function(input, lambd = 0.5) {
  torch_softshrink(input, lambd)
}

nnf_softsign <- function(input) {
  input / (input$abs() + 1)
}

nnf_tanhshrink <- function(input) {
  input - input$tanh()
}

nnf_threshold <- function(input, threshold, value, inplace = FALSE) {
  if (inplace)
    torch_threshold_(input, threshold, value)
  else
    torch_threshold(input, threshold, value)
}

nnf_threshold_ <- function(input, threshold, value) {
  nnf_threshold(input, threshold, value, TRUE)
}

nnf_multi_head_attention_forward <- function(
  query,                           # type: Tensor
  key,                             # type: Tensor
  value,                           # type: Tensor
  embed_dim_to_check,              # type: int
  num_heads,                       # type: int
  in_proj_weight,                  # type: Tensor
  in_proj_bias,                    # type: Tensor
  bias_k,                          # type: Optional[Tensor]
  bias_v,                          # type: Optional[Tensor]
  add_zero_attn,                   # type: bool
  dropout_p,                       # type: float
  out_proj_weight,                 # type: Tensor
  out_proj_bias,                   # type: Tensor
  training=TRUE,                   # type: bool
  key_padding_mask=NULL,           # type: Optional[Tensor]
  need_weights=TRUE,               # type: bool
  attn_mask=NULL,                  # type: Optional[Tensor]
  use_separate_proj_weight=FALSE,  # type: bool
  q_proj_weight=NULL,              # type: Optional[Tensor]
  k_proj_weight=NULL,              # type: Optional[Tensor]
  v_proj_weight=NULL,              # type: Optional[Tensor]
  static_k=NULL,                   # type: Optional[Tensor]
  static_v=NULL                    # type: Optional[Tensor])
) {
  
  o <- query$size()
  tgt_len <- o[[1]]; bsz <- o[[2]]; embed_dim <- o[[3]];
  
  
  head_dim <- floor(embed_dim / num_heads) 
  
  scaling <- head_dim^(-0.5)
  
  if (!use_separate_proj_weight) {
    
    if (torch_equal(query, key) & torch_equal(key, value)) {
      # self-attention
      o <- nnf_linear(query, in_proj_weight, in_proj_bias)$chunk(3, dim = -1)
      q <- o[[1]]; k <- o[[2]]; v <- o[[3]]
    } else if (torch_equal(key, value)) {
      # encoder-decoder attention
      #             # This is inline in_proj function with in_proj_weight and in_proj_bias
      b_ <- in_proj_bias
      start_ <- 1
      end_ <- embed_dim
      w_ <- in_proj_weight[start_:end_,]
      
      if (!is.null(b_)) {
        b_ <- b_[start_:end_] 
      }
      
      q <- nnf_linear(query, w_, b_)
      
      if (is.null(key)) {
        k <- NULL
        v <- NULL
      } else {
        b_ <- in_proj_bias
        start_ <- embed_dim
        end_ <- NULL
        w_ <- in_proj_weight[start_:N, ]
        if (!is.null(b_)) {
          b_ <- b_[start_:N]
          o <- nnf_linear(key, w_, b_)$chunk(2, dim = -1)
          k <- o[[1]]; v <- o[[2]]
        }
        
      }
      
    } else {
      
      # This is inline in_proj function with in_proj_weight and in_proj_bias
      b_ <- in_proj_bias
      start_ <- 0
      end_ <- embed_dim
      w_ <- in_proj_weight[start_:end_, ]
      if (!is.null(b_))
        b_ <- b_[start_:end_]
      q <- nnf_linear(query, w_, b_)
      
      
      # This is inline in_proj function with in_proj_weight and in_proj_bias
      b_ <- in_proj_bias
      start_ <- embed_dim
      end_ <- embed_dim * 2
      w_ <- in_proj_weight[start_:end_,]
      if (!is.null(b_))
        b_ <- b_[start_:end_]
      k <- nnf_linear(key, w_, b_)
      
      # This is inline in_proj function with in_proj_weight and in_proj_bias
      b_ <- in_proj_bias
      start_ <- embed_dim * 2
      end_ <- NULL
      w_ <- in_proj_weight[start_:N,]
      if (!is.null(b_)) {
        b_ <- b_[start_:N]
      }
      v <- nnf_linear(value, w_, b_)
      
    }
    
  } else {
    
    if (!is.null(in_proj_bias)) {
      q <- nnf_linear(query, q_proj_weight, in_proj_bias[1:embed_dim])
      k <- nnf_linear(key, k_proj_weight, in_proj_bias[embed_dim:(embed_dim * 2)])
      v <- nnf_linear(value, v_proj_weight, in_proj_bias[(embed_dim*2):N])
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
      
      if (!is.null(attn_mask))
        attn_mask <- nnf_pad(attn_mask, c(0,1))
      
      if (!is.null(key_padding_mask))
        key_padding_mask <- nnf_pad(key_padding_mask, c(0,1))
      
    }
  }
  
  q <- q$contiguous()$view(tgt_len, bsz * num_heads, head_dim)$transpose(0,1)
  
  if (!is.null(k))
    k <- k$contiguous()$view(-1, bsz * num_heads, head_dim)$transpose(0,1)
  
  if (!is.null(v))
    v <- v$contiguous()$view(-1, bsz * num_heads, head_dim)$transpose(0,1)
  
  
  if (!is.null(static_k))
    k <- static_k
  
  if (!is.null(static_v))
    v <- static_v
  
  src_len <- k$size(1)   
  
  if (add_zero_attn) {
    src_len <- src_len + 1
    k_size <- k$size()
    k <- torch_cat(list(k, torch_zeros(append(list(k_size[1], 1), k_size[3:length(k_size)]),
                                       dtype = k$dtype, device = k$device)), dim = 1)
    v_size
    k <- torch_cat(list(k, torch_zeros(append(list(v_size[1], 1), v_size[3:length(v_size)]),
                                       dtype = v$dtype, device = v$device)), dim = 1)
    
    if (!is.null(attn_mask)) {
      attn_mask <- nnf_pad(attn_mask, list(0,1))
    }
    
    if (!is.null(key_padding_mask)) {
      key_padding_mask <- nnf_pad(key_padding_mask, list(0,1))
    }
    
  }
  
  attn_output_weights <- torch_bmm(q, k$transpose(1, 2))
  
  if (!is.null(attn_mask)) {
    if (attn_mask$dtype == torch_bool()) {
      attn_output_weights$masked_fill_(attn_mask, -Inf)
    } else {
      attn_output_weights <- attn_output_weights + attn_mask
    }
  }
  
  if (!is.null(key_padding_mask)) {
    attn_output_weights <- attn_output_weights$view(vsz, num_heads, tgt_len, src_len)
    attn_output_weights <- attn_output_weights$masked_fill(
      key_padding_mask$unsqueeze(1)$unsqueeze(2),
      -Inf
    )
    attn_output_weights <- attn_output_weights$view(bsz * num_heads, tgt_len, 
                                                    src_len)
  }
  
  attn_output_weights <- nnf_softmax(attn_output_weights, dim=-1)
  attn_output_weights <- nnf_dropout(attn_output_weights, p=dropout_p, 
                                     training=training)
  
  attn_output <- torch_bmm(attn_output_weights, v)
  attn_output <- attn_output$transpose(0, 1)$contiguous()$view(tgt_len, bsz, embed_dim)
  attn_output <- nnf_linear(attn_output, out_proj_weight, out_proj_bias)
  
  if (need_weights) {
    attn_output_weights <- attn_output_weights$view(bsz, num_heads, tgt_len, 
                                                    src_len)
    return(list(attn_output, attn_output_weights$sum(dim = 1)/num_heads))
  } else {
    return(list(attn_output, NULL))
  }
}
