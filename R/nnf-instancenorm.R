#' Instance_norm
#'
#' Applies Instance Normalization for each channel in each data sample in a
#' batch.
#'
#' @param input the input tensor
#' @param running_mean the running_mean tensor
#' @param running_var the running var tensor
#' @param weight the weight tensor
#' @param bias the bias tensor
#' @param use_input_stats whether to use input stats
#' @param momentum a double for the momentum
#' @param eps an eps double for numerical stability
#'
#' @export
nnf_instance_norm <- function(input, running_mean = NULL, running_var = NULL,
                              weight = NULL, bias = NULL, use_input_stats = TRUE,
                              momentum = 0.1, eps = 1e-5) {
  torch_instance_norm(
    input, weight, bias, running_mean, running_var,
    use_input_stats, momentum, eps, FALSE # TODO backend_cudnn_enabled)
  )
}
