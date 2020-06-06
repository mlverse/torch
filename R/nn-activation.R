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
  forward = function() {
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
