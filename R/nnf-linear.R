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
    if (!is.null(bias)) {
      output <- output + bias
    }
    ret <- output
  }

  ret
}

#' Bilinear
#'
#' Applies a bilinear transformation to the incoming data:
#' \eqn{y = x_1 A x_2 + b}
#'
#'
#' @param input1 \eqn{(N, *, H_{in1})} where \eqn{H_{in1}=\mbox{in1\_features}}
#'  and \eqn{*} means any number of additional dimensions.
#'  All but the last dimension of the inputs should be the same.
#' @param input2 \eqn{(N, *, H_{in2})} where \eqn{H_{in2}=\mbox{in2\_features}}
#' @param weight \eqn{(\mbox{out\_features}, \mbox{in1\_features},
#'  \mbox{in2\_features})}
#' @param bias \eqn{(\mbox{out\_features})}
#'
#' @return
#'
#' output \eqn{(N, *, H_{out})} where \eqn{H_{out}=\mbox{out\_features}}
#' and all but the last dimension are the same shape as the input.
#'
#' @export
nnf_bilinear <- function(input1, input2, weight, bias = NULL) {
  torch_bilinear(input1, input2, weight, bias)
}
