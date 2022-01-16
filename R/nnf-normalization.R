#' Normalize
#'
#' Performs \eqn{L_p} normalization of inputs over specified dimension.
#'
#' For a tensor `input` of sizes \eqn{(n_0, ..., n_{dim}, ..., n_k)}, each
#' \eqn{n_{dim}} -element vector \eqn{v} along dimension `dim` is transformed as
#'
#' \deqn{
#'         v = \frac{v}{\max(\Vert v \Vert_p, \epsilon)}.
#' }
#'
#' With the default arguments it uses the Euclidean norm over vectors along
#' dimension \eqn{1} for normalization.
#'
#' @param input input tensor of any shape
#' @param p (float) the exponent value in the norm formulation. Default: 2
#' @param dim (int) the dimension to reduce. Default: 1
#' @param eps (float) small value to avoid division by zero. Default: 1e-12
#' @param out (Tensor, optional) the output tensor. If `out` is used, this                            operation won't be differentiable.
#'
#'
#' @export
nnf_normalize <- function(input, p = 2, dim = 2, eps = 1e-12, out = NULL) {
  if (is.null(out)) {
    denom <- input$norm(p, dim, keepdim = TRUE)$clamp_min(eps)$expand_as(input)
    return(input / denom)
  } else {
    denom <- input$norm(p, dim, keepdim = TRUE)$clamp_min_(eps)$expand_as(input)
    return(torch_div_out(out, input, denom))
  }
}

#' Layer_norm
#'
#' Applies Layer Normalization for last certain number of dimensions.
#'
#' @param input the input tensor
#' @param normalized_shape input shape from an expected input of size. If a single
#'   integer is used, it is treated as a singleton list, and this module will normalize
#'   over the last dimension which is expected to be of that specific size.
#' @param weight the weight tensor
#' @param bias the bias tensor
#' @param eps a value added to the denominator for numerical stability. Default: 1e-5
#'
#' @export
nnf_layer_norm <- function(input, normalized_shape, weight = NULL, bias = NULL,
                           eps = 1e-5) {
  torch_layer_norm(
    input = input,
    normalized_shape = normalized_shape,
    weight = weight,
    bias = bias,
    eps = eps,
    cudnn_enable = FALSE
  )
}

#' Local_response_norm
#'
#' Applies local response normalization over an input signal composed of
#' several input planes, where channels occupy the second dimension.
#' Applies normalization across channels.
#'
#'
#' @param input the input tensor
#' @param size amount of neighbouring channels used for normalization
#' @param alpha multiplicative factor. Default: 0.0001
#' @param beta exponent. Default: 0.75
#' @param k additive factor. Default: 1
#'
#' @export
nnf_local_response_norm <- function(input, size, alpha = 1e-4, beta = 0.75, k = 1) {
  dim <- input$dim()
  div <- input$mul(input)$unsqueeze(1)

  if (dim == 3) {
    div <- nnf_pad(div, c(0, 0, as.integer(size / 2), as.integer((size - 1) / 2)))
    div <- nnf_avg_pool2d(div, c(size, 1), stride = 1)$squeeze(1)
  } else {
    sizes <- input$size()
    div <- div$view(c(sizes[1], 1, sizes[2], sizes[3], -1))
    div <- nnf_pad(div, c(0, 0, 0, 0, as.integer(size / 2), as.integer((size - 1) / 2)))
    div <- nnf_avg_pool3d(div, c(size, 1, 1), stride = 1)$squeeze(1)
    div <- div$view(sizes)
  }

  div <- div$mul(alpha)$add(k)$pow(beta)
  input / div
}

#' Group_norm
#'
#' Applies Group Normalization for last certain number of dimensions.
#'
#' @param input the input tensor
#' @param num_groups number of groups to separate the channels into
#' @param weight the weight tensor
#' @param bias the bias tensor
#' @param eps a value added to the denominator for numerical stability. Default: 1e-5
#'
#' @export
nnf_group_norm <- function(input, num_groups, weight = NULL, bias = NULL,
                           eps = 1e-5) {
  torch_group_norm(input,
    num_groups = num_groups, weight = weight,
    bias = bias, eps = eps # TODO ,cudnn_enabled = backends_cudnn_enabled
  )
}
