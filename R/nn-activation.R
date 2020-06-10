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
#' m <- nn_hardswish()
#' input <- torch_randn(2)
#' output <- m(input)
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

