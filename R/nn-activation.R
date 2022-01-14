#' @include nn.R
NULL

#' Threshoold module
#'
#' Thresholds each element of the input Tensor.
#'
#' Threshold is defined as:
#' \deqn{
#'   y =
#'   \left\{ \begin{array}{ll}
#'   x, &\mbox{ if } x > \mbox{threshold} \\
#'   \mbox{value}, &\mbox{ otherwise }
#'   \end{array}
#'   \right.
#' }
#'
#' @param threshold The value to threshold at
#' @param value The value to replace with
#' @param inplace can optionally do the operation in-place. Default: `FALSE`
#'
#' @section Shape:
#' - Input: \eqn{(N, *)} where `*` means, any number of additional
#' dimensions
#' - Output: \eqn{(N, *)}, same shape as the input
#'
#' @examples
#' m <- nn_threshold(0.1, 20)
#' input <- torch_randn(2)
#' output <- m(input)
#' @export
nn_threshold <- nn_module(
  "nn_threshold",
  initialize = function(threshold, value, inplace = FALSE) {
    self$threshold <- threshold
    self$value <- value
    self$inplace <- inplace
  },
  forward = function(input) {
    nnf_threshold(input, self$threshold, self$value, self$inplace)
  }
)

#' ReLU module
#'
#' Applies the rectified linear unit function element-wise
#' \deqn{\mbox{ReLU}(x) = (x)^+ = \max(0, x)}
#'
#' @param inplace can optionally do the operation in-place. Default: `FALSE`
#'
#' @section Shape:
#'
#' - Input: \eqn{(N, *)} where `*` means, any number of additional
#'   dimensions
#' - Output: \eqn{(N, *)}, same shape as the input
#'
#' @examples
#' m <- nn_relu()
#' input <- torch_randn(2)
#' m(input)
#' @export
nn_relu <- nn_module(
  "nn_relu",
  initialize = function(inplace = FALSE) {
    self$inplace <- inplace
  },
  forward = function(input) {
    nnf_relu(input, self$inplace)
  }
)

#' RReLU module
#'
#' Applies the randomized leaky rectified liner unit function, element-wise,
#' as described in the paper:
#'
#' `Empirical Evaluation of Rectified Activations in Convolutional Network`.
#'
#' The function is defined as:
#'
#' \deqn{
#' \mbox{RReLU}(x) =
#' \left\{ \begin{array}{ll}
#' x & \mbox{if } x \geq 0 \\
#' ax & \mbox{ otherwise }
#' \end{array}
#' \right.
#' }
#'
#' where \eqn{a} is randomly sampled from uniform distribution
#' \eqn{\mathcal{U}(\mbox{lower}, \mbox{upper})}.
#' See: https://arxiv.org/pdf/1505.00853.pdf
#'
#' @param lower lower bound of the uniform distribution. Default: \eqn{\frac{1}{8}}
#' @param upper upper bound of the uniform distribution. Default: \eqn{\frac{1}{3}}
#' @param inplace can optionally do the operation in-place. Default: `FALSE`
#'
#' @section Shape:
#'
#' - Input: \eqn{(N, *)} where `*` means, any number of additional
#' dimensions
#' - Output: \eqn{(N, *)}, same shape as the input
#'
#' @examples
#' m <- nn_rrelu(0.1, 0.3)
#' input <- torch_randn(2)
#' m(input)
#' @export
nn_rrelu <- nn_module(
  "nn_rrelu",
  initialize = function(lower = 1 / 8, upper = 1 / 3, inplace = FALSE) {
    self$lower <- lower
    self$upper <- upper
    self$inplace <- inplace
  },
  forward = function(input) {
    nnf_rrelu(input, self$lower, self$upper, self$training, self$inplace)
  }
)

#' Hardtanh module
#'
#' Applies the HardTanh function element-wise
#' HardTanh is defined as:
#'
#' \deqn{
#' \mbox{HardTanh}(x) = \left\{ \begin{array}{ll}
#'   1 & \mbox{ if } x > 1 \\
#'   -1 & \mbox{ if } x < -1 \\
#'   x & \mbox{ otherwise } \\
#' \end{array}
#' \right.
#' }
#'
#' The range of the linear region :math:`[-1, 1]` can be adjusted using
#' `min_val` and `max_val`.
#'
#' @param min_val minimum value of the linear region range. Default: -1
#' @param max_val maximum value of the linear region range. Default: 1
#' @param inplace can optionally do the operation in-place. Default: `FALSE`
#'
#' @section Shape:
#'
#' - Input: \eqn{(N, *)} where `*` means, any number of additional
#'   dimensions
#' - Output: \eqn{(N, *)}, same shape as the input
#'
#' @examples
#' m <- nn_hardtanh(-2, 2)
#' input <- torch_randn(2)
#' output <- m(input)
#' @export
nn_hardtanh <- nn_module(
  "nn_hardtanh",
  initialize = function(min_val = -1, max_val = 1, inplace = FALSE) {
    self$min_val <- min_val
    self$max_val <- max_val
    self$inplace <- inplace
  },
  forward = function(input) {
    nnf_hardtanh(input, self$min_val, self$max_val, self$inplace)
  }
)

#' ReLu6 module
#'
#' Applies the element-wise function:
#'
#' \deqn{
#'   \mbox{ReLU6}(x) = \min(\max(0,x), 6)
#' }
#'
#' @param inplace can optionally do the operation in-place. Default: `FALSE`
#'
#' @section Shape:
#'
#' - Input: \eqn{(N, *)} where `*` means, any number of additional
#'   dimensions
#' - Output: \eqn{(N, *)}, same shape as the input
#'
#' @examples
#' m <- nn_relu6()
#' input <- torch_randn(2)
#' output <- m(input)
#' @export
nn_relu6 <- nn_module(
  "nn_relu6",
  inherit = nn_hardtanh,
  initialize = function(inplace = FALSE) {
    super$initialize(0, 6, inplace)
  }
)


#' Sigmoid module
#'
#' Applies the element-wise function:
#'
#' \deqn{
#'   \mbox{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}
#' }
#'
#' @section Shape:
#'
#' - Input: \eqn{(N, *)} where `*` means, any number of additional
#'   dimensions
#' - Output: \eqn{(N, *)}, same shape as the input
#'
#' @examples
#' m <- nn_sigmoid()
#' input <- torch_randn(2)
#' output <- m(input)
#' @export
nn_sigmoid <- nn_module(
  "nn_sigmoid",
  initialize = function() {},
  forward = function(input) {
    torch_sigmoid(input)
  }
)

#' Hardsigmoid module
#'
#' Applies the element-wise function:
#'
#' \deqn{
#' \mbox{Hardsigmoid}(x) = \left\{ \begin{array}{ll}
#'   0 & \mbox{if~} x \le -3, \\
#'   1 & \mbox{if~} x \ge +3, \\
#'   x / 6 + 1 / 2 & \mbox{otherwise}
#' \end{array}
#' \right.
#' }
#'
#' @section Shape:
#' - Input: \eqn{(N, *)} where `*` means, any number of additional
#' dimensions
#' - Output: \eqn{(N, *)}, same shape as the input
#'
#' @examples
#' m <- nn_hardsigmoid()
#' input <- torch_randn(2)
#' output <- m(input)
#' @export
nn_hardsigmoid <- nn_module(
  "nn_hardsigmoid",
  initialize = function() {},
  forward = function(input) {
    nnf_hardsigmoid(input)
  }
)

#' Tanh module
#'
#' Applies the element-wise function:
#'
#' \deqn{
#'   \mbox{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}
#' }
#'
#' @section Shape:
#' - Input: \eqn{(N, *)} where `*` means, any number of additional
#'   dimensions
#' - Output: \eqn{(N, *)}, same shape as the input
#'
#' @examples
#' m <- nn_tanh()
#' input <- torch_randn(2)
#' output <- m(input)
#' @export
nn_tanh <- nn_module(
  "nn_tanh",
  initialize = function() {},
  forward = function(input) {
    torch_tanh(input)
  }
)

#' Hardswish module
#'
#' Applies the hardswish function, element-wise, as described in the paper:
#' [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
#'
#'
#' \deqn{ \mbox{Hardswish}(x) = \left\{
#'   \begin{array}{ll}
#'   0 & \mbox{if } x \le -3, \\
#'   x & \mbox{if } x \ge +3, \\
#'   x \cdot (x + 3)/6 & \mbox{otherwise}
#'   \end{array}
#'   \right. }
#'
#' @section Shape:
#'
#' - Input: \eqn{(N, *)} where `*` means, any number of additional
#' dimensions
#' - Output: \eqn{(N, *)}, same shape as the input
#'
#' @examples
#' \dontrun{
#' m <- nn_hardswish()
#' input <- torch_randn(2)
#' output <- m(input)
#' }
#'
#' @export
nn_hardswish <- nn_module(
  "nn_hardswish",
  initialize = function() {},
  forward = function(input) {
    nnf_hardswish(input)
  }
)

#' ELU module
#'
#' Applies the element-wise function:
#'
#' \deqn{
#'   \mbox{ELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x) - 1))
#' }
#'
#' @param alpha the \eqn{\alpha} value for the ELU formulation. Default: 1.0
#' @param inplace can optionally do the operation in-place. Default: `FALSE`
#'
#' @section Shape:
#'
#' - Input: \eqn{(N, *)} where `*` means, any number of additional
#' dimensions
#' - Output: \eqn{(N, *)}, same shape as the input
#'
#' @examples
#' m <- nn_elu()
#' input <- torch_randn(2)
#' output <- m(input)
#' @export
nn_elu <- nn_module(
  "nn_elu",
  initialize = function(alpha = 1, inplace = FALSE) {
    self$alpha <- alpha
    self$inplace <- inplace
  },
  forward = function(input) {
    nnf_elu(input, self$alpha, self$inplace)
  }
)

#' CELU module
#'
#' Applies the element-wise function:
#'
#' \deqn{
#'   \mbox{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))
#' }
#'
#' More details can be found in the paper
#' [Continuously Differentiable Exponential Linear Units](https://arxiv.org/abs/1704.07483).
#'
#' @param alpha the \eqn{\alpha} value for the CELU formulation. Default: 1.0
#' @param inplace can optionally do the operation in-place. Default: `FALSE`
#'
#' @section Shape:
#'
#' - Input: \eqn{(N, *)} where `*` means, any number of additional
#'   dimensions
#' - Output: \eqn{(N, *)}, same shape as the input
#'
#' @examples
#' m <- nn_celu()
#' input <- torch_randn(2)
#' output <- m(input)
#' @export
nn_celu <- nn_module(
  "nn_celu",
  initialize = function(alpha = 1, inplace = FALSE) {
    self$alpha <- alpha
    self$inplace <- inplace
  },
  forward = function(input) {
    nnf_celu(input, self$alpha, self$inplace)
  }
)

#' SELU module
#'
#' Applied element-wise, as:
#'
#' \deqn{
#'   \mbox{SELU}(x) = \mbox{scale} * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)))
#' }
#'
#' with \eqn{\alpha = 1.6732632423543772848170429916717} and
#' \eqn{\mbox{scale} = 1.0507009873554804934193349852946}.
#'
#' More details can be found in the paper
#' [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515).
#'
#' @param inplace (bool, optional): can optionally do the operation in-place. Default: `FALSE`
#'
#' @section Shape:
#' - Input: \eqn{(N, *)} where `*` means, any number of additional
#'   dimensions
#' - Output: \eqn{(N, *)}, same shape as the input
#'
#' @examples
#' m <- nn_selu()
#' input <- torch_randn(2)
#' output <- m(input)
#' @export
#
nn_selu <- nn_module(
  "nn_selu",
  initialize = function(inplace = FALSE) {
    self$inplace <- inplace
  },
  forward = function(input) {
    nnf_selu(input, self$inplace)
  }
)

#' GLU module
#'
#' Applies the gated linear unit function
#' \eqn{{GLU}(a, b)= a \otimes \sigma(b)} where \eqn{a} is the first half
#' of the input matrices and \eqn{b} is the second half.
#'
#' @param dim (int): the dimension on which to split the input. Default: -1
#'
#' @section Shape:
#'
#' - Input: \eqn{(\ast_1, N, \ast_2)} where `*` means, any number of additional
#'   dimensions
#' - Output: \eqn{(\ast_1, M, \ast_2)} where \eqn{M=N/2}
#'
#' @examples
#' m <- nn_glu()
#' input <- torch_randn(4, 2)
#' output <- m(input)
#' @export
nn_glu <- nn_module(
  "nn_glue",
  initialize = function(dim = -1) {
    self$dim <- dim
  },
  forward = function(input) {
    nnf_glu(input, self$dim)
  }
)

#' GELU module
#'
#' Applies the Gaussian Error Linear Units function:
#'   \deqn{\mbox{GELU}(x) = x * \Phi(x)}
#'
#' where \eqn{\Phi(x)} is the Cumulative Distribution Function for Gaussian Distribution.
#'
#' @section Shape:
#' - Input: \eqn{(N, *)} where `*` means, any number of additional
#'   dimensions
#' - Output: \eqn{(N, *)}, same shape as the input
#'
#' @examples
#' m <- nn_gelu()
#' input <- torch_randn(2)
#' output <- m(input)
#' @export
nn_gelu <- nn_module(
  "nn_gelu",
  initialize = function() {},
  forward = function(input) {
    nnf_gelu(input)
  }
)

#' Hardshwink module
#'
#' Applies the hard shrinkage function element-wise:
#'
#' \deqn{
#'   \mbox{HardShrink}(x) =
#'   \left\{ \begin{array}{ll}
#' x, & \mbox{ if } x > \lambda \\
#' x, & \mbox{ if } x < -\lambda \\
#' 0, & \mbox{ otherwise }
#' \end{array}
#' \right.
#' }
#'
#' @param lambd the \eqn{\lambda} value for the Hardshrink formulation. Default: 0.5
#'
#' @section Shape:
#'
#' - Input: \eqn{(N, *)} where `*` means, any number of additional
#'   dimensions
#' - Output: \eqn{(N, *)}, same shape as the input
#'
#' @examples
#' m <- nn_hardshrink()
#' input <- torch_randn(2)
#' output <- m(input)
#' @export
nn_hardshrink <- nn_module(
  "nn_hardshrink",
  initialize = function(lambd = 0.5) {
    self$lambd <- lambd
  },
  forward = function(input) {
    nnf_hardshrink(input, self$lambd)
  }
)

#' LeakyReLU module
#'
#' Applies the element-wise function:
#'
#' \deqn{
#'   \mbox{LeakyReLU}(x) = \max(0, x) + \mbox{negative\_slope} * \min(0, x)
#' }
#' or
#'
#' \deqn{
#'   \mbox{LeakyRELU}(x) =
#'   \left\{ \begin{array}{ll}
#' x, & \mbox{ if } x \geq 0 \\
#' \mbox{negative\_slope} \times x, & \mbox{ otherwise }
#' \end{array}
#' \right.
#' }
#'
#' @param negative_slope Controls the angle of the negative slope. Default: 1e-2
#' @param inplace can optionally do the operation in-place. Default: `FALSE`
#'
#' @section Shape:
#'
#' - Input: \eqn{(N, *)} where `*` means, any number of additional
#'   dimensions
#' - Output: \eqn{(N, *)}, same shape as the input
#'
#' @examples
#' m <- nn_leaky_relu(0.1)
#' input <- torch_randn(2)
#' output <- m(input)
#' @export
nn_leaky_relu <- nn_module(
  "nn_leaky_relu",
  initialize = function(negative_slope = 1e-2, inplace = FALSE) {
    self$negative_slope <- negative_slope
    self$inplace <- inplace
  },
  forward = function(input) {
    nnf_leaky_relu(input, self$negative_slope, self$inplace)
  }
)

#' LogSigmoid module
#'
#' Applies the element-wise function:
#' \deqn{
#'   \mbox{LogSigmoid}(x) = \log\left(\frac{ 1 }{ 1 + \exp(-x)}\right)
#'  }
#'
#' @section Shape:
#'
#' - Input: \eqn{(N, *)} where `*` means, any number of additional
#' dimensions
#' - Output: \eqn{(N, *)}, same shape as the input
#'
#' @examples
#' m <- nn_log_sigmoid()
#' input <- torch_randn(2)
#' output <- m(input)
#' @export
nn_log_sigmoid <- nn_module(
  "nn_log_sigmoid",
  initialize = function() {},
  forward = function(input) {
    nnf_logsigmoid(input)
  }
)

#' Softplus module
#'
#' Applies the element-wise function:
#' \deqn{
#'   \mbox{Softplus}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x))
#' }
#'
#' SoftPlus is a smooth approximation to the ReLU function and can be used
#' to constrain the output of a machine to always be positive.
#' For numerical stability the implementation reverts to the linear function
#' when \eqn{input \times \beta > threshold}.
#'
#' @param beta the \eqn{\beta} value for the Softplus formulation. Default: 1
#' @param threshold values above this revert to a linear function. Default: 20
#'
#' @section Shape:
#'
#' - Input: \eqn{(N, *)} where `*` means, any number of additional
#' dimensions
#' - Output: \eqn{(N, *)}, same shape as the input
#'
#' @examples
#' m <- nn_softplus()
#' input <- torch_randn(2)
#' output <- m(input)
#' @export
nn_softplus <- nn_module(
  "nn_softplus",
  initialize = function(beta = 1, threshold = 20) {
    self$beta <- beta
    self$threshold <- threshold
  },
  forward = function(input) {
    nnf_softplus(input, self$beta, self$threshold)
  }
)

#' Softshrink module
#'
#' Applies the soft shrinkage function elementwise:
#'
#' \deqn{
#'   \mbox{SoftShrinkage}(x) =
#'   \left\{ \begin{array}{ll}
#' x - \lambda, & \mbox{ if } x > \lambda \\
#' x + \lambda, & \mbox{ if } x < -\lambda \\
#' 0, & \mbox{ otherwise }
#' \end{array}
#' \right.
#' }
#'
#' @param lambd the \eqn{\lambda} (must be no less than zero) value for the Softshrink formulation. Default: 0.5
#'
#' @section Shape:
#' - Input: \eqn{(N, *)} where `*` means, any number of additional
#' dimensions
#' - Output: \eqn{(N, *)}, same shape as the input
#'
#' @examples
#' m <- nn_softshrink()
#' input <- torch_randn(2)
#' output <- m(input)
#' @export
nn_softshrink <- nn_module(
  "nn_softshrink",
  initialize = function(lambd = 0.5) {
    self$lambd <- lambd
  },
  forward = function(input) {
    nnf_softshrink(input, self$lambd)
  }
)

#' MultiHead attention
#'
#' Allows the model to jointly attend to information
#' from different representation subspaces.
#' See reference: Attention Is All You Need
#'
#' \deqn{
#'   \mbox{MultiHead}(Q, K, V) = \mbox{Concat}(head_1,\dots,head_h)W^O
#' \mbox{where} head_i = \mbox{Attention}(QW_i^Q, KW_i^K, VW_i^V)
#' }
#'
#' @param embed_dim total dimension of the model.
#' @param num_heads parallel attention heads.
#' @param dropout a Dropout layer on attn_output_weights. Default: 0.0.
#' @param bias add bias as module parameter. Default: True.
#' @param add_bias_kv add bias to the key and value sequences at dim=0.
#' @param add_zero_attn add a new batch of zeros to the key and
#'   value sequences at dim=1.
#' @param kdim total number of features in key. Default: `NULL`
#' @param vdim total number of features in value. Default: `NULL`.
#'   Note: if kdim and vdim are `NULL`, they will be set to embed_dim such that
#'   query, key, and value have the same number of features.
#'
#' @section Shape:
#'
#' Inputs:
#'
#' - query: \eqn{(L, N, E)} where L is the target sequence length, N is the batch size, E is
#' the embedding dimension.
#' - key: \eqn{(S, N, E)}, where S is the source sequence length, N is the batch size, E is
#' the embedding dimension.
#' - value: \eqn{(S, N, E)} where S is the source sequence length, N is the batch size, E is
#' the embedding dimension.
#' - key_padding_mask: \eqn{(N, S)} where N is the batch size, S is the source sequence length.
#'   If a ByteTensor is provided, the non-zero positions will be ignored while the position
#'   with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
#'   value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
#' - attn_mask: 2D mask \eqn{(L, S)} where L is the target sequence length, S is the source sequence length.
#'   3D mask \eqn{(N*num_heads, L, S)} where N is the batch size, L is the target sequence length,
#'   S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
#'   positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
#'   while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
#'   is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
#'   is provided, it will be added to the attention weight.
#'
#' Outputs:
#'
#' - attn_output: \eqn{(L, N, E)} where L is the target sequence length, N is
#'   the batch size, E is the embedding dimension.
#' - attn_output_weights:
#'   - if ``avg_weights`` is ``TRUE`` (the default), the output attention
#'     weights are averaged over the attention heads, giving a tensor of shape
#'     \eqn{(N, L, S)} where N is the batch size, L is the target sequence
#'     length, S is the source sequence length.
#'   - if ``avg_weights`` is ``FALSE``, the attention weight tensor is output
#'     as-is, with shape \eqn{(N, H, L, S)}, where H is the number of attention
#'     heads.
#'
#' @examples
#' \dontrun{
#' multihead_attn <- nn_multihead_attention(embed_dim, num_heads)
#' out <- multihead_attn(query, key, value)
#' attn_output <- out[[1]]
#' attn_output_weights <- out[[2]]
#' }
#'
#' @export
nn_multihead_attention <- nn_module(
  "nn_multihead_attention",
  initialize = function(embed_dim, num_heads, dropout = 0., bias = TRUE, add_bias_kv = FALSE,
                        add_zero_attn = FALSE, kdim = NULL, vdim = NULL) {
    self$embed_dim <- embed_dim

    if (!is.null(kdim)) {
      self$kdim <- kdim
    } else {
      self$kdim <- embed_dim
    }

    if (!is.null(vdim)) {
      self$vdim <- vdim
    } else {
      self$vdim <- embed_dim
    }

    self$qkv_same_embed_dim_ <- self$kdim == embed_dim && self$vdim == embed_dim

    self$num_heads <- num_heads
    self$dropout <- dropout
    self$head_dim <- embed_dim %/% num_heads

    if (!self$qkv_same_embed_dim_) {
      self$q_proj_weight <- nn_parameter(torch_empty(embed_dim, embed_dim))
      self$k_proj_weight <- nn_parameter(torch_empty(embed_dim, self$kdim))
      self$v_proj_weight <- nn_parameter(torch_empty(embed_dim, self$vdim))
      self$register_parameter("in_proj_weight", NULL)
    } else {
      self$in_proj_weight <- nn_parameter(torch_empty(3 * embed_dim, embed_dim))
      self$register_parameter("q_proj_weight", NULL)
      self$register_parameter("k_proj_weight", NULL)
      self$register_parameter("v_proj_weight", NULL)
    }

    if (bias) {
      self$in_proj_bias <- nn_parameter(torch_empty(3 * embed_dim))
    } else {
      self$register_parameter("in_proj_bias", NULL)
    }

    self$out_proj <- nn_linear(embed_dim, embed_dim, bias = bias)

    if (add_bias_kv) {
      self$bias_k <- nn_parameter(torch_empty(1, 1, embed_dim))
      self$bias_v <- nn_parameter(torch_empty(1, 1, embed_dim))
    } else {
      self$bias_k <- NULL
      self$bias_v <- NULL
    }

    self$add_zero_attn <- add_zero_attn

    self$reset_parameters()
  },
  reset_parameters = function() {
    if (self$qkv_same_embed_dim_) {
      nn_init_xavier_uniform_(self$in_proj_weight)
    } else {
      nn_init_xavier_uniform_(self$q_proj_weight)
      nn_init_xavier_uniform_(self$k_proj_weight)
      nn_init_xavier_uniform_(self$v_proj_weight)
    }

    if (!is.null(self$in_proj_bias)) {
      nn_init_constant_(self$in_proj_bias, 0)
      nn_init_constant_(self$out_proj$bias, 0)
    }

    if (!is.null(self$bias_k)) {
      nn_init_xavier_normal_(self$bias_k)
    }

    if (!is.null(self$bias_v)) {
      nn_init_xavier_normal_(self$bias_v)
    }
  },
  forward = function(query, key, value, key_padding_mask = NULL,
                     need_weights = TRUE, attn_mask = NULL, avg_weights = TRUE) {
    if (!self$qkv_same_embed_dim_) {
      nnf_multi_head_attention_forward(
        query, key, value, self$embed_dim, self$num_heads,
        self$in_proj_weight, self$in_proj_bias,
        self$bias_k, self$bias_v, self$add_zero_attn,
        self$dropout, self$out_proj$weight, self$out_proj$bias,
        training = self$training,
        key_padding_mask = key_padding_mask, need_weights = need_weights,
        attn_mask = attn_mask, use_separate_proj_weight = TRUE,
        q_proj_weight = self$q_proj_weight, k_proj_weight = self$k_proj_weight,
        v_proj_weight = self$v_proj_weight
      )
    } else {
      nnf_multi_head_attention_forward(
        query, key, value, self$embed_dim, self$num_heads,
        self$in_proj_weight, self$in_proj_bias,
        self$bias_k, self$bias_v, self$add_zero_attn,
        self$dropout, self$out_proj$weight, self$out_proj$bias,
        training = self$training,
        key_padding_mask = key_padding_mask, need_weights = need_weights,
        attn_mask = attn_mask, avg_weights = avg_weights
      )
    }
  }
)

#' PReLU module
#'
#' Applies the element-wise function:
#' \deqn{
#'   \mbox{PReLU}(x) = \max(0,x) + a * \min(0,x)
#' }
#' or
#' \deqn{
#'   \mbox{PReLU}(x) =
#'   \left\{ \begin{array}{ll}
#' x, & \mbox{ if } x \geq 0 \\
#' ax, & \mbox{ otherwise }
#' \end{array}
#' \right.
#' }
#'
#' Here \eqn{a} is a learnable parameter. When called without arguments, `nn.prelu()` uses a single
#' parameter \eqn{a} across all input channels. If called with `nn_prelu(nChannels)`,
#' a separate \eqn{a} is used for each input channel.
#'
#' @note weight decay should not be used when learning \eqn{a} for good performance.
#'
#' @note Channel dim is the 2nd dim of input. When input has dims < 2, then there is
#'   no channel dim and the number of channels = 1.
#'
#' @param num_parameters (int): number of \eqn{a} to learn.
#'   Although it takes an int as input, there is only two values are legitimate:
#'   1, or the number of channels at input. Default: 1
#' @param init (float): the initial value of \eqn{a}. Default: 0.25
#'
#' @section Shape:
#'
#' - Input: \eqn{(N, *)} where `*` means, any number of additional
#'   dimensions
#' - Output: \eqn{(N, *)}, same shape as the input
#'
#' @section Attributes:
#'
#' - weight (Tensor): the learnable weights of shape (`num_parameters`).
#'
#' @examples
#' m <- nn_prelu()
#' input <- torch_randn(2)
#' output <- m(input)
#' @export
nn_prelu <- nn_module(
  "nn_prelu",
  initialize = function(num_parameters = 1, init = 0.25) {
    self$num_parameters <- num_parameters
    self$weight <- nn_parameter(torch_empty(num_parameters)$fill_(init))
  },
  forward = function(input) {
    nnf_prelu(input, self$weight)
  }
)

#' Softsign module
#'
#' Applies the element-wise function:
#' \deqn{
#'   \mbox{SoftSign}(x) = \frac{x}{ 1 + |x|}
#' }
#'
#' @section Shape:
#'
#' - Input: \eqn{(N, *)} where `*` means, any number of additional
#' dimensions
#' - Output: \eqn{(N, *)}, same shape as the input
#'
#' @examples
#' m <- nn_softsign()
#' input <- torch_randn(2)
#' output <- m(input)
#' @export
nn_softsign <- nn_module(
  "nn_softsign",
  initialize = function() {},
  forward = function(input) {
    nnf_softsign(input)
  }
)

#' Tanhshrink module
#'
#' Applies the element-wise function:
#'
#' \deqn{
#'   \mbox{Tanhshrink}(x) = x - \tanh(x)
#' }
#'
#' @section Shape:
#' - Input: \eqn{(N, *)} where `*` means, any number of additional
#' dimensions
#' - Output: \eqn{(N, *)}, same shape as the input
#'
#' @examples
#' m <- nn_tanhshrink()
#' input <- torch_randn(2)
#' output <- m(input)
#' @export
nn_tanhshrink <- nn_module(
  "nn_tanhshrink",
  initialize = function() {},
  forward = function(input) {
    nnf_tanhshrink(input)
  }
)

#' Softmin
#'
#' Applies the Softmin function to an n-dimensional input Tensor
#' rescaling them so that the elements of the n-dimensional output Tensor
#' lie in the range `[0, 1]` and sum to 1.
#' Softmin is defined as:
#'
#' \deqn{
#'   \mbox{Softmin}(x_{i}) = \frac{\exp(-x_i)}{\sum_j \exp(-x_j)}
#' }
#'
#' @section Shape:
#'
#' - Input: \eqn{(*)} where `*` means, any number of additional
#' dimensions
#' - Output: \eqn{(*)}, same shape as the input
#'
#' @param dim (int): A dimension along which Softmin will be computed (so every slice
#'    along dim will sum to 1).
#'
#' @return
#' a Tensor of the same dimension and shape as the input, with
#' values in the range `[0, 1]`.
#'
#' @examples
#' m <- nn_softmin(dim = 1)
#' input <- torch_randn(2, 2)
#' output <- m(input)
#' @export
nn_softmin <- nn_module(
  "nn_softmin",
  initialize = function(dim) {
    self$dim <- dim
  },
  forward = function(input) {
    nnf_softmin(input, self$dim)
  }
)

#' Softmax module
#'
#' Applies the Softmax function to an n-dimensional input Tensor
#' rescaling them so that the elements of the n-dimensional output Tensor
#' lie in the range `[0,1]` and sum to 1.
#' Softmax is defined as:
#'
#' \deqn{
#'   \mbox{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
#' }
#'
#' When the input Tensor is a sparse tensor then the unspecifed
#' values are treated as `-Inf`.
#'
#' @section Shape:
#'
#' - Input: \eqn{(*)} where `*` means, any number of additional
#' dimensions
#' - Output: \eqn{(*)}, same shape as the input
#'
#' @return:
#' a Tensor of the same dimension and shape as the input with
#' values in the range `[0, 1]`
#'
#' @param dim (int): A dimension along which Softmax will be computed (so every slice
#'                                                                along dim will sum to 1).
#' @note
#' This module doesn't work directly with NLLLoss,
#' which expects the Log to be computed between the Softmax and itself.
#' Use `LogSoftmax` instead (it's faster and has better numerical properties).
#'
#' @examples
#' m <- nn_softmax(1)
#' input <- torch_randn(2, 3)
#' output <- m(input)
#' @export
nn_softmax <- nn_module(
  "nn_softmax",
  initialize = function(dim) {
    self$dim <- dim
  },
  forward = function(input) {
    nnf_softmax(input, self$dim)
  }
)

#' Softmax2d module
#'
#' Applies SoftMax over features to each spatial location.
#' When given an image of `Channels x Height x Width`, it will
#' apply `Softmax` to each location \eqn{(Channels, h_i, w_j)}
#'
#' @section Shape:
#' - Input: \eqn{(N, C, H, W)}
#' - Output: \eqn{(N, C, H, W)} (same shape as input)
#'
#' @return
#' a Tensor of the same dimension and shape as the input with
#' values in the range `[0, 1]`
#'
#' @examples
#' m <- nn_softmax2d()
#' input <- torch_randn(2, 3, 12, 13)
#' output <- m(input)
#' @export
nn_softmax2d <- nn_module(
  "nn_softmax2d",
  initialize = function() {},
  forward = function(input) {
    nnf_softmax(input, dim = 1)
  }
)

#' LogSoftmax module
#'
#' Applies the \eqn{\log(\mbox{Softmax}(x))} function to an n-dimensional
#' input Tensor. The LogSoftmax formulation can be simplified as:
#'
#' \deqn{
#'   \mbox{LogSoftmax}(x_{i}) = \log\left(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} \right)
#' }
#'
#' @section Shape:
#'
#' - Input: \eqn{(*)} where `*` means, any number of additional
#' dimensions
#' - Output: \eqn{(*)}, same shape as the input
#'
#' @param dim (int): A dimension along which LogSoftmax will be computed.
#'
#' @return
#' a Tensor of the same dimension and shape as the input with
#' values in the range [-inf, 0)
#'
#' @examples
#' m <- nn_log_softmax(1)
#' input <- torch_randn(2, 3)
#' output <- m(input)
#' @export
nn_log_softmax <- nn_module(
  "nn_log_softmax",
  initialize = function(dim) {
    self$dim <- dim
  },
  forward = function(input) {
    nnf_log_softmax(input, self$dim)
  }
)

#' Sparsemax activation
#'
#' Sparsemax activation module.
#'
#' @details
#' The SparseMax activation is described in
#' ['From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification'](https://arxiv.org/abs/1602.02068)
#' The implementation is based on [aced125/sparsemax](https://github.com/aced125/sparsemax/tree/master/sparsemax)
#'
#' @param dim The dimension over which to apply the sparsemax function. (-1)
#'
#' @export
nn_contrib_sparsemax <- nn_module(
  "nn_contrib_sparsemax",
  initialize = function(dim = -1) {
    self$dim <- dim
  },
  forward = function(input) {
    nnf_contrib_sparsemax(input, self$dim)
  }
)
