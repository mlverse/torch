
#' Pad
#'
#' Pads tensor.
#'
#' @section Padding size:
#'
#' The padding size by which to pad some dimensions of `input`
#' are described starting from the last dimension and moving forward.
#' \eqn{\left\lfloor\frac{\mbox{len(pad)}}{2}\right\rfloor} dimensions
#' of ``input`` will be padded.
#' For example, to pad only the last dimension of the input tensor, then
#' `pad` has the form
#' \eqn{(\mbox{padding\_left}, \mbox{padding\_right})};
#' to pad the last 2 dimensions of the input tensor, then use
#' \eqn{(\mbox{padding\_left}, \mbox{padding\_right},}
#' \eqn{\mbox{padding\_top}, \mbox{padding\_bottom})};
#' to pad the last 3 dimensions, use
#' \eqn{(\mbox{padding\_left}, \mbox{padding\_right},}
#' \eqn{\mbox{padding\_top}, \mbox{padding\_bottom}}
#' \eqn{\mbox{padding\_front}, \mbox{padding\_back})}.
#'
#' @section Padding mode:
#'
#' See `nn_constant_pad_2d`, `nn_reflection_pad_2d`, and
#' `nn_replication_pad_2d` for concrete examples on how each of the
#' padding modes works. Constant padding is implemented for arbitrary dimensions.
#  Replicate padding is implemented for padding the last 3 dimensions of 5D input
#' tensor, or the last 2 dimensions of 4D input tensor, or the last dimension of
#' 3D input tensor. Reflect padding is only implemented for padding the last 2
#' dimensions of 4D input tensor, or the last dimension of 3D input tensor.
#'
#' @param input (Tensor) N-dimensional tensor
#' @param pad (tuple) m-elements tuple, where \eqn{\frac{m}{2} \leq} input dimensions
#'   and \eqn{m} is even.
#' @param mode 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
#' @param value fill value for 'constant' padding. Default: 0.
#'
#' @export
nnf_pad <- function(input, pad, mode = "constant", value = 0) {
  if (mode == "constant") {
    return(torch_constant_pad_nd(input, pad, value))
  } else {
    if (input$dim() == 3) {
      if (mode == "reflect") {
        return(torch_reflection_pad1d(input, pad))
      }

      if (mode == "replicate") {
        return(torch_replication_pad1d(input, pad))
      }

      if (mode == "circular") {
        return(nnf_pad_circular(input, pad))
      }

      not_implemented_error()
    }

    if (input$dim() == 4) {
      if (mode == "reflect") {
        return(torch_reflection_pad2d(input, pad))
      }

      if (mode == "replicate") {
        return(torch_replication_pad2d(input, pad))
      }

      if (mode == "circular") {
        return(nnf_pad_circular(input, pad))
      }

      not_implemented_error()
    }

    if (input$dim() == 5) {
      if (mode == "reflect") {
        not_implemented_error()
      }

      if (mode == "replicate") {
        return(torch_replication_pad3d(input, pad))
      }

      if (mode == "circular") {
        return(nnf_pad_circular(input, pad))
      }
    }
  }

  not_implemented_error("Only 3D, 4D, 5D padding with non-constant padding are supported for now")
}
