#' @include nn.R
NULL

#' Threshoold module
#' 
#' Thresholds each element of the input Tensor.
#'
#' Threshold is defined as:
#' \deqn{
#'   y = 
#'   \begin{cases}
#'   x, &\text{ if } x > \text{threshold} \\
#'   \text{value}, &\text{ otherwise }
#'   \end{cases}
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
#'
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
#' \deqn{\text{ReLU}(x) = (x)^+ = \max(0, x)}
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
#' 
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
#' 
#' \text{RReLU}(x) =
#' \begin{cases}
#' x & \text{if } x \geq 0 \\
#' ax & \text{ otherwise }
#' \end{cases}
#' 
#' }
#' 
#' where \eqn{a} is randomly sampled from uniform distribution
#' \eqn{\mathcal{U}(\text{lower}, \text{upper})}.
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
#' 
#' @export
nn_rrelu <- nn_module(
  "nn_rrelu",
  initialize = function(lower = 1/8, upper =1/3, inplace = FALSE) {
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
#' \text{HardTanh}(x) = \begin{cases}
#'   1 & \text{ if } x > 1 \\
#'   -1 & \text{ if } x < -1 \\
#'   x & \text{ otherwise } \\
#' \end{cases}
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
#' 
#' @export
nn_hardtanh <- nn_module(
  "nn_hardtanh",
  initialize = function(min_val = -1, max_val = 1, inplace = FALSE) {
    self$min_val = min_val
    self$max_val = max_val
    self$inplace = inplace
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
#'   \text{ReLU6}(x) = \min(\max(0,x), 6)
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
#' 
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
#'   \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}
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
#' 
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
#' \text{Hardsigmoid}(x) = \begin{cases}
#'   0 & \text{if~} x \le -3, \\
#'   1 & \text{if~} x \ge +3, \\
#'   x / 6 + 1 / 2 & \text{otherwise}
#' \end{cases}
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
#' 
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
#'   \text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}
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
#' 
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
#' \deqn{
#'   \text{Hardswish}(x) = \begin{cases}
#' 0 & \text{if~} x \le -3, \\
#' x & \text{if~} x \ge +3, \\
#' x \cdot (x + 3) /6 & \text{otherwise}
#' \end{cases}
#' }
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
#'   \text{ELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x) - 1))
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
#' output <-  m(input)
#' 
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
#'   \text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))
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
#' 
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
#'   \text{SELU}(x) = \text{scale} * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)))
#' }
#' 
#' with \eqn{\alpha = 1.6732632423543772848170429916717} and
#' \eqn{\text{scale} = 1.0507009873554804934193349852946}.
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
#' 
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
#' 
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
#'   \deqn{\text{GELU}(x) = x * \Phi(x)}
#' 
#' where \eqn{\Phi(x)} is the Cumulative Distribution Function for Gaussian Distribution.
#' 
#' @section Shape:
#' - Input: \eqn{(N, *)} where `*` means, any number of additional
#'   dimensions
#' - Output: \eqn{(N, *)}, same shape as the input
#' 
#' @examples
#' m = nn_gelu()
#' input <- torch_randn(2)
#' output <- m(input)
#' 
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
#'   \text{HardShrink}(x) =
#'   \begin{cases}
#' x, & \text{ if } x > \lambda \\
#' x, & \text{ if } x < -\lambda \\
#' 0, & \text{ otherwise }
#' \end{cases}
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
#' 
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
#'   \text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)
#' }
#' or
#' 
#' \deqn{
#'   \text{LeakyRELU}(x) =
#'   \begin{cases}
#' x, & \text{ if } x \geq 0 \\
#' \text{negative\_slope} \times x, & \text{ otherwise }
#' \end{cases}
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
#' 
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
#'   \text{LogSigmoid}(x) = \log\left(\frac{ 1 }{ 1 + \exp(-x)}\right)
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
#' 
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
#'   \text{Softplus}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x))
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
#' 
#' @export
nn_softplus <- nn_module(
  "nn_softplus",
  initialize = function(beta=1, threshold=20) {
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
#'   \text{SoftShrinkage}(x) =
#'   \begin{cases}
#' x - \lambda, & \text{ if } x > \lambda \\
#' x + \lambda, & \text{ if } x < -\lambda \\
#' 0, & \text{ otherwise }
#' \end{cases}
#' 
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
#' 
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
#'   \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
#' \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
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
#' - attn_output: \eqn{(L, N, E)} where L is the target sequence length, N is the batch size,
#'   E is the embedding dimension.
#' - attn_output_weights: \eqn{(N, L, S)} where N is the batch size,
#'   L is the target sequence length, S is the source sequence length.
#'   
#' @examples
#' \dontrun{
#' multihead_attn = nn_multihead_attention(embed_dim, num_heads)
#' out <- multihead_attn(query, key, value)
#' attn_output <- out[[1]]
#' attn_output_weights <- out[[2]]
#' }
#' 
#' @export
nn_multihead_attention <- nn_module(
  "nn_multihead_attention",
  initialize = function(embed_dim, num_heads, dropout=0., bias=TRUE, add_bias_kv=FALSE, 
                        add_zero_attn=FALSE, kdim=NULL, vdim=NULL) {
    
    self$embed_dim <- embed_dim
    
    if (!is.null(kdim))
      self$kdim <- kdim
    else
      self$kdim <- embed_dim
    
    if (!is.null(vdim))
      self$vdim <- vdim
    else
      self$vdim <- embed_dim
    
    self$qkv_same_embed_dim_ <- self$kdim == embed_dim && self$vdim == embed_dim
    
    self$num_heads <- num_heads
    self$dropout <- dropout
    self$head_dim <- embed_dim %/% num_heads
    
    if (!self$qkv_same_embed_dim_) {
      self$q_proj_weight <- nn_parameter(torch_empty(embed_dim, embed_dim))
      self$k_proj_weight = nn_parameter(torch_empty(embed_dim, self$kdim))
      self$v_proj_weight = nn_parameter(torch_empty(embed_dim, self$vdim))
      self$register_parameter('in_proj_weight', NULL)
    } else {
      self$in_proj_weight = nn_parameter(torch_empty(3 * embed_dim, embed_dim))
      self$register_parameter('q_proj_weight', NULL)
      self$register_parameter('k_proj_weight', NULL)
      self$register_parameter('v_proj_weight', NULL)
    }
    
    if (bias)
      self$in_proj_bias <- nn_parameter(torch_empty(3 * embed_dim))
    else
      self$register_parameter("in_proj_bias", NULL)
    
    self$out_proj <- nn_linear(embed_dim, embed_dim, bias=bias)
    
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
    
    if (self$qkv_same_embed_dim) {
      nn_init_xavier_uniform_(self$in_proj_weight)
    } else {
      nn_init_xavier_uniform_(self$q_proj_weight)
      nn_init_xavier_uniform_(self$k_proj_weight)
      nn_init_xavier_uniform_(self$v_proj_weight)
    }
    
    if (!is.null(self$in_proj_bias)) {
      nn_init_constant_(self$in_proj_bias, 0)
      nn_init_constant_(self$out_proj_bias, 0)
    }
    
    if (!is.null(self$bias_k)) {
      nn_init_xavier_normal_(self$bias_k)
    }
    
    if (!is.null(self$bias_v)) {
      nn_init_xavier_normal_(self$bias_v)
    }
    
  },
  forward = function(query, key, value, key_padding_mask=NULL,
                     need_weights=TRUE, attn_mask=NULL) {
    if (!self$qkv_same_embed_dim_) {
      nnf_multi_head_attention_forward(
        query, key, value, self$embed_dim, self$num_heads,
        self$in_proj_weight, self$in_proj_bias,
        self$bias_k, self$bias_v, self$add_zero_attn,
        self$dropout, self$out_proj.weight, self$out_proj.bias,
        training=self$training,
        key_padding_mask=key_padding_mask, need_weights=need_weights,
        attn_mask=attn_mask, use_separate_proj_weight=TRUE,
        q_proj_weight=self$q_proj_weight, k_proj_weight=self$k_proj_weight,
        v_proj_weight=self$v_proj_weight)
    } else {
      nnf_multi_head_attention_forward(
        query, key, value, self$embed_dim, self$num_heads,
        self$in_proj_weight, self$in_proj_bias,
        self$bias_k, self$bias_v, self$add_zero_attn,
        self$dropout, self$out_proj.weight, self$out_proj.bias,
        training=self$training,
        key_padding_mask=key_padding_mask, need_weights=need_weights,
        attn_mask=attn_mask)
    }
  }
)