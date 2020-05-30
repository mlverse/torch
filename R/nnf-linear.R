#' Linear
#'
#' Applies a linear transformation to the incoming data: \eqn{y = xA^T + b}.
#' 
#' @param input \eqn{(N, *, in\_features)} where `*` means any number of
#'    additional dimensions
#' @param weight \eqn{(out\_features, in\_features)} the weights tensor.
#' @param bias optional tensor \eqn{(out\_features)}
#'
#'
#' @export
nnf_linear <- function(input, weight, bias = NULL) {
  
  if (input$dim() == 2 && !is.null(bias)) {
    ret <- torch_addmm(bias, input, weight$t())
  } else {
    output <- input$matmul(weight$t())
    if (!is.null(bias))
      output <- output + bias
    ret <- output
  }
  
  ret
}