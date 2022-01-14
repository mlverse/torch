#' Pairwise distance
#'
#' Computes the batchwise pairwise distance between vectors \eqn{v_1}, \eqn{v_2}
#' using the p-norm:
#'
#' \deqn{
#'  \Vert x \Vert _p = \left( \sum_{i=1}^n  \vert x_i \vert ^ p \right) ^ {1/p}.
#' }
#'
#' @param p (real): the norm degree. Default: 2
#' @param eps (float, optional): Small value to avoid division by zero.
#'    Default: 1e-6
#' @param keepdim (bool, optional): Determines whether or not to keep the vector dimension.
#'    Default: FALSE
#'
#' @section Shape:
#' - Input1: \eqn{(N, D)} where `D = vector dimension`
#' - Input2: \eqn{(N, D)}, same shape as the Input1
#' - Output: \eqn{(N)}. If `keepdim` is `TRUE`, then \eqn{(N, 1)}.
#'
#' @examples
#' pdist <- nn_pairwise_distance(p = 2)
#' input1 <- torch_randn(100, 128)
#' input2 <- torch_randn(100, 128)
#' output <- pdist(input1, input2)
#' @export
nn_pairwise_distance <- nn_module(
  "nn_pairwise_distance",
  initialize = function(p = 2, eps = 1e-6, keepdim = FALSE) {
    self$norm <- p
    self$eps <- eps
    self$keepdim <- keepdim
  },
  forward = function(x1, x2) {
    nnf_pairwise_distance(x1, x2,
      p = self$norm, eps = self$eps,
      keepdim = self$keepdim
    )
  }
)
