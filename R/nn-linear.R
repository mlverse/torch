#' @include nn.R
NULL

#' Identity module
#'
#' A placeholder identity operator that is argument-insensitive.
#'
#' @param ... any arguments (unused)
#'
#' @examples
#' m <- nn_identity(54, unused_argument1 = 0.1, unused_argument2 = FALSE)
#' input <- torch_randn(128, 20)
#' output <- m(input)
#' print(output$size())
#' @export
nn_identity <- nn_module(
  "nn_identity",
  initialize = function(...) {},
  forward = function(input) {
    input
  }
)

#' Linear module
#'
#' Applies a linear transformation to the incoming data: `y = xA^T + b`
#'
#' @param in_features size of each input sample
#' @param out_features size of each output sample
#' @param bias If set to `FALSE`, the layer will not learn an additive bias.
#'   Default: `TRUE`
#'
#' @section Shape:
#'
#' - Input: `(N, *, H_in)` where `*` means any number of
#'   additional dimensions and `H_in = in_features`.
#' - Output: `(N, *, H_out)` where all but the last dimension
#'   are the same shape as the input and :math:`H_out = out_features`.
#'
#' @section Attributes:
#'
#' - weight: the learnable weights of the module of shape
#'   `(out_features, in_features)`. The values are
#'   initialized from \eqn{U(-\sqrt{k}, \sqrt{k})}s, where
#'   \eqn{k = \frac{1}{\mbox{in\_features}}}
#' - bias: the learnable bias of the module of shape \eqn{(\mbox{out\_features})}.
#'   If `bias` is `TRUE`, the values are initialized from
#'   \eqn{\mathcal{U}(-\sqrt{k}, \sqrt{k})} where
#'   \eqn{k = \frac{1}{\mbox{in\_features}}}
#'
#' @examples
#' m <- nn_linear(20, 30)
#' input <- torch_randn(128, 20)
#' output <- m(input)
#' print(output$size())
#' @export
nn_linear <- nn_module(
  "nn_linear",
  initialize = function(in_features, out_features, bias = TRUE) {
    self$in_features <- in_features
    self$out_feature <- out_features

    self$weight <- nn_parameter(torch_empty(out_features, in_features))
    if (bias) {
      self$bias <- nn_parameter(torch_empty(out_features))
    } else {
      self$bias <- NULL
    }

    self$reset_parameters()
  },
  reset_parameters = function() {
    nn_init_kaiming_uniform_(self$weight, a = sqrt(5))
    if (!is.null(self$bias)) {
      fans <- nn_init_calculate_fan_in_and_fan_out(self$weight)
      bound <- 1 / sqrt(fans[[1]])
      nn_init_uniform_(self$bias, -bound, bound)
    }
  },
  forward = function(input) {
    nnf_linear(input, self$weight, self$bias)
  }
)

#' Bilinear module
#'
#' Applies a bilinear transformation to the incoming data
#' \eqn{y = x_1^T A x_2 + b}
#'
#' @param in1_features size of each first input sample
#' @param in2_features size of each second input sample
#' @param out_features size of each output sample
#' @param bias If set to `FALSE`, the layer will not learn an additive bias.
#'   Default: `TRUE`
#'
#' @section Shape:
#'
#' - Input1: \eqn{(N, *, H_{in1})} \eqn{H_{in1}=\mbox{in1\_features}} and
#'   \eqn{*} means any number of additional dimensions. All but the last
#'   dimension of the inputs should be the same.
#' - Input2: \eqn{(N, *, H_{in2})} where \eqn{H_{in2}=\mbox{in2\_features}}.
#' - Output: \eqn{(N, *, H_{out})} where \eqn{H_{out}=\mbox{out\_features}}
#'   and all but the last dimension are the same shape as the input.
#'
#' @section Attributes:
#'
#' - weight: the learnable weights of the module of shape
#'   \eqn{(\mbox{out\_features}, \mbox{in1\_features}, \mbox{in2\_features})}.
#'   The values are initialized from \eqn{\mathcal{U}(-\sqrt{k}, \sqrt{k})}, where
#'   \eqn{k = \frac{1}{\mbox{in1\_features}}}
#' - bias: the learnable bias of the module of shape \eqn{(\mbox{out\_features})}.
#'   If `bias` is `TRUE`, the values are initialized from
#'   \eqn{\mathcal{U}(-\sqrt{k}, \sqrt{k})}, where
#'   \eqn{k = \frac{1}{\mbox{in1\_features}}}
#'
#' @examples
#' m <- nn_bilinear(20, 30, 50)
#' input1 <- torch_randn(128, 20)
#' input2 <- torch_randn(128, 30)
#' output <- m(input1, input2)
#' print(output$size())
#' @export
nn_bilinear <- nn_module(
  "nn_bilinear",
  initialize = function(in1_features, in2_features, out_features, bias = TRUE) {
    self$in1_features <- in1_features
    self$in2_features <- in2_features
    self$out_features <- out_features

    self$weight <- nn_parameter(torch_empty(out_features, in1_features, in2_features))

    if (bias) {
      self$bias <- nn_parameter(torch_empty(out_features))
    } else {
      self$bias <- NULL
    }

    self$reset_parameters()
  },
  reset_parameters = function() {
    bound <- 1 / sqrt(tail(self$weight$size(), 1))
    nn_init_uniform_(self$weight, -bound, bound)

    if (!is.null(self$bias)) {
      nn_init_uniform_(self$bias, -bound, bound)
    }
  },
  forward = function(input1, input2) {
    nnf_bilinear(input1, input2, self$weight, self$bias)
  }
)
