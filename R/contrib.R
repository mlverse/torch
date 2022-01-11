#' Contrib sort vertices
#'
#' Based on the implementation from [Rotated_IoU](https://github.com/lilanxiao/Rotated_IoU)
#'
#' @note This function does not make part of the official torch API.
#' @details All tensors should be on a CUDA device so this function can be used.
#'
#' @param vertices A Tensor with the vertices.
#' @param mask A tensors containing the masks.
#' @param num_valid A integer tensors.
#'
#' @examples
#' if (cuda_is_available()) {
#'   v <- torch_randn(8, 1024, 24, 2)$cuda()
#'   mean <- torch_mean(v, dim = 2, keepdim = TRUE)
#'   v <- v - mean
#'   m <- (torch_rand(8, 1024, 24) > 0.8)$cuda()
#'   nv <- torch_sum(m$to(dtype = torch_int()), dim = -1)$to(dtype = torch_int())$cuda()
#'   result <- contrib_sort_vertices(v, m, nv)
#' }
#' @export
contrib_sort_vertices <- function(vertices, mask, num_valid) {
  cpp_contrib_torch_sort_vertices(vertices, mask, num_valid)
}
