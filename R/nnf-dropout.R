#' Dropout
#'
#' During training, randomly zeroes some of the elements of the input
#' tensor with probability `p` using samples from a Bernoulli
#' distribution.
#'
#' @param input the input tensor
#' @param p probability of an element to be zeroed. Default: 0.5
#' @param training apply dropout if is `TRUE`. Default: `TRUE`
#' @param inplace If set to `TRUE`, will do this operation in-place.
#'   Default: `FALSE`
#'
#' @export
nnf_dropout <- function(input, p = 0.5, training = TRUE, inplace = FALSE) {
  if (inplace) {
    torch_dropout_(input, p, training)
  } else {
    torch_dropout(input, p, training)
  }
}

#' Dropout2d
#'
#' Randomly zero out entire channels (a channel is a 2D feature map,
#' e.g., the \eqn{j}-th channel of the \eqn{i}-th sample in the
#' batched input is a 2D tensor \eqn{input[i, j]}) of the input tensor).
#' Each channel will be zeroed out independently on every forward call with
#' probability `p` using samples from a Bernoulli distribution.
#'
#' @inheritParams nnf_dropout
#' @param p probability of a channel to be zeroed. Default: 0.5
#' @param training apply dropout if is `TRUE`. Default: `TRUE`.
#' @param inplace If set to `TRUE`, will do this operation in-place.
#'   Default: `FALSE`
#'
#' @export
nnf_dropout2d <- function(input, p = 0.5, training = TRUE, inplace = FALSE) {
  if (inplace) {
    torch_feature_dropout_(input, p, training)
  } else {
    torch_feature_dropout(input, p, training)
  }
}

#' Dropout3d
#'
#' Randomly zero out entire channels (a channel is a 3D feature map,
#' e.g., the \eqn{j}-th channel of the \eqn{i}-th sample in the
#' batched input is a 3D tensor \eqn{input[i, j]}) of the input tensor).
#' Each channel will be zeroed out independently on every forward call with
#' probability `p` using samples from a Bernoulli distribution.
#'
#' @inheritParams nnf_dropout2d
#'
#' @export
nnf_dropout3d <- function(input, p = 0.5, training = TRUE, inplace = FALSE) {
  if (inplace) {
    torch_feature_dropout_(input, p, training)
  } else {
    torch_feature_dropout(input, p, training)
  }
}

#' Alpha_dropout
#'
#' Applies alpha dropout to the input.
#'
#' @inheritParams nnf_dropout
#'
#' @export
nnf_alpha_dropout <- function(input, p = 0.5, training = FALSE, inplace = FALSE) {
  if (inplace) {
    torch_alpha_dropout_(input, p, training)
  } else {
    torch_alpha_dropout(input, p, training)
  }
}
