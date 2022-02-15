#' @include nn.R
NULL

#' Flattens a contiguous range of dims into a tensor.
#'
#' For use with [nn_sequential].
#'
#' @section Shape:
#' - Input: `(*, S_start,..., S_i, ..., S_end, *)`,
#'           where `S_i` is the size at dimension `i` and `*` means any
#'           number of dimensions including none.
#' - Output: `(*, S_start*...*S_i*...S_end, *)`.
#'
#' @param start_dim first dim to flatten (default = 2).
#' @param end_dim last dim to flatten (default = -1).
#'
#' @examples
#' input <- torch_randn(32, 1, 5, 5)
#' m <- nn_flatten()
#' m(input)
#' @seealso [nn_unflatten]
#' @export
nn_flatten <- nn_module(
  "nn_flatten",
  initialize = function(start_dim = 2, end_dim = -1) {
    self$start_dim <- start_dim
    self$end_dim <- end_dim
  },
  forward = function(input) {
    input$flatten(start_dim = self$start_dim, end_dim = self$end_dim)
  }
)

#' Unflattens a tensor dim expanding it to a desired shape.
#' For use with [[nn_sequential].
#'
#' @param dim Dimension to be unflattened
#' @param unflattened_size New shape of the unflattened dimension
#'
#' @examples
#' input <- torch_randn(2, 50)
#'
#' m <- nn_sequential(
#'   nn_linear(50, 50),
#'   nn_unflatten(2, c(2, 5, 5))
#' )
#' output <- m(input)
#' output$size()
#' @export
nn_unflatten <- nn_module(
  "nn_unflatten",
  initialize = function(dim, unflattened_size) {
    self$dim <- dim
    self$unflattened_size <- unflattened_size
  },
  forward = function(input) {
    input$unflatten(self$dim, self$unflattened_size)
  }
)
