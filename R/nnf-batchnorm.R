#' Batch_norm
#'
#' Applies Batch Normalization for each channel across a batch of data.
#'
#' @param input input tensor
#' @param running_mean the running_mean tensor
#' @param running_var the running_var tensor
#' @param weight the weight tensor
#' @param bias  the bias tensor
#' @param training bool wether it's training. Default: FALSE
#' @param momentum the value used for the `running_mean` and `running_var` computation.
#'  Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1
#' @param eps a value added to the denominator for numerical stability. Default: 1e-5
#'
#' @export
nnf_batch_norm <- function(input, running_mean, running_var, weight = NULL, bias = NULL,
                           training = FALSE, momentum = 0.1, eps = 1e-5) {
  torch_batch_norm(
    input = input, weight = weight, bias = bias, running_mean = running_mean,
    running_var = running_var, training = training, momentum = momentum,
    eps = eps, cudnn_enabled = backends_cudnn_enabled()
  )
}
