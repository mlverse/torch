#' Cosine_similarity
#'
#' Returns cosine similarity between x1 and x2, computed along dim.
#'
#' \deqn{
#'     \mbox{similarity} = \frac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}
#' }
#'
#'
#' @param x1 (Tensor) First input.
#' @param x2 (Tensor) Second input (of size matching x1).
#' @param dim (int, optional) Dimension of vectors. Default: 2
#' @param eps (float, optional) Small value to avoid division by zero.
#'   Default: 1e-8
#'
#' @export
nnf_cosine_similarity <- function(x1, x2, dim = 2, eps = 1e-8) {
  torch_cosine_similarity(x1 = x1, x2 = x2, dim = dim, eps = eps)
}

#' Pairwise_distance
#'
#' Computes the batchwise pairwise distance between vectors using the p-norm.
#'
#' @inheritParams nnf_cosine_similarity
#' @param keepdim Determines whether or not to keep the vector dimension. Default: False
#' @param p the norm degree. Default: 2
#'
#'
#' @export
nnf_pairwise_distance <- function(x1, x2, p = 2, eps = 1e-6, keepdim = FALSE) {
  torch_pairwise_distance(x1, x2, p, eps, keepdim)
}

#' Pdist
#'
#' Computes the p-norm distance between every pair of row vectors in the input.
#' This is identical to the upper triangular portion, excluding the diagonal, of
#' `torch_norm(input[:, None] - input, dim=2, p=p)`. This function will be faster
#' if the rows are contiguous.
#'
#' If input has shape \eqn{N \times M} then the output will have shape
#' \eqn{\frac{1}{2} N (N - 1)}.
#'
#'
#' @param input input tensor of shape \eqn{N \times M}.
#' @param p p value for the p-norm distance to calculate between each vector pair
#'   \eqn{\in [0, \infty]}.
#'
#'
#' @export
nnf_pdist <- function(input, p = 2) {
  torch_pdist(input, p)
}
