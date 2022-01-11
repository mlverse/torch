#' Embedding
#'
#' A simple lookup table that looks up embeddings in a fixed dictionary and size.
#'
#' This module is often used to retrieve word embeddings using indices.
#' The input to the module is a list of indices, and the embedding matrix,
#' and the output is the corresponding word embeddings.
#'
#' @param input (LongTensor) Tensor containing indices into the embedding matrix
#' @param weight (Tensor) The embedding matrix with number of rows equal to the
#'   maximum possible index + 1, and number of columns equal to the embedding size
#' @param padding_idx (int, optional) If given, pads the output with the embedding
#'   vector at `padding_idx` (initialized to zeros) whenever it encounters the index.
#' @param max_norm (float, optional) If given, each embedding vector with norm larger
#'   than `max_norm` is renormalized to have norm `max_norm`. Note: this will modify
#'   `weight` in-place.
#' @param norm_type (float, optional) The p of the p-norm to compute for the `max_norm`
#'   option. Default `2`.
#' @param scale_grad_by_freq (boolean, optional) If given, this will scale gradients
#'   by the inverse of frequency of the words in the mini-batch. Default `FALSE`.
#' @param sparse (bool, optional) If `TRUE`, gradient w.r.t. `weight` will be a
#'   sparse tensor. See Notes under `nn_embedding` for more details regarding
#'   sparse gradients.
#'
#' @export
nnf_embedding <- function(input, weight, padding_idx = NULL, max_norm = NULL, norm_type = 2,
                          scale_grad_by_freq = FALSE, sparse = FALSE) {
  if (is.null(padding_idx)) {
    padding_idx <- -1
  }

  if (!is.null(max_norm)) {
    input <- input$contiguous()
    with_no_grad({
      torch_embedding_renorm_(weight, input, max_norm, norm_type)
    })
  }

  torch_embedding(
    weight = weight, indices = input, padding_idx = padding_idx,
    scale_grad_by_freq = scale_grad_by_freq, sparse = sparse
  )
}

#' Embedding_bag
#'
#' Computes sums, means or maxes of `bags` of embeddings, without instantiating the
#' intermediate embeddings.
#'
#' @param input (LongTensor) Tensor containing bags of indices into the embedding matrix
#' @param weight (Tensor) The embedding matrix with number of rows equal to the
#'   maximum possible index + 1, and number of columns equal to the embedding size
#' @param offsets (LongTensor, optional) Only used when `input` is 1D. `offsets`
#'   determines the starting index position of each bag (sequence) in `input`.
#' @param max_norm (float, optional) If given, each embedding vector with norm
#'   larger than `max_norm` is renormalized to have norm `max_norm`.
#'   Note: this will modify `weight` in-place.
#' @param norm_type (float, optional) The ``p`` in the ``p``-norm to compute for the
#'   `max_norm` option. Default ``2``.
#' @param scale_grad_by_freq (boolean, optional) if given, this will scale gradients
#'   by the inverse of frequency of the words in the mini-batch. Default `FALSE`.                                            Note: this option is not supported when ``mode="max"``.
#' @param mode (string, optional) ``"sum"``, ``"mean"`` or ``"max"``. Specifies
#'   the way to reduce the bag. Default: 'mean'
#' @param sparse (bool, optional) if `TRUE`, gradient w.r.t. `weight` will be a
#'   sparse tensor. See Notes under `nn_embedding` for more details regarding
#'   sparse gradients. Note: this option is not supported when `mode="max"`.
#' @param per_sample_weights (Tensor, optional) a tensor of float / double weights,
#'  or NULL to indicate all weights should be taken to be 1. If specified,
#'  `per_sample_weights` must have exactly the same shape as input and is treated
#'  as having the same `offsets`, if those are not `NULL`.
#' @param include_last_offset (bool, optional) if `TRUE`, the size of offsets is
#'   equal to the number of bags + 1.
#'
#' @export
nnf_embedding_bag <- function(input, weight, offsets = NULL, max_norm = NULL,
                              norm_type = 2, scale_grad_by_freq = FALSE,
                              mode = "mean", sparse = FALSE, per_sample_weights = NULL,
                              include_last_offset = FALSE) {
  if (input$dim() == 2) {
    input <- input$reshape(-1)
    if (!is.null(per_sample_weights)) {
      per_sample_weights <- per_sample_weights$reshape(-1)
    }
  }

  if (mode == "sum") {
    mode_enum <- 0
  } else if (mode == "mean") {
    mode_enum <- 1
  } else if (mode == "max") {
    mode_enum <- 2
  }

  if (!is.null(max_norm)) {
    input <- input$contiguous()
    with_no_grad({
      torch_embedding_renorm_(weight, input, max_norm, norm_type)
    })
  }

  ret <- torch_embedding_bag(
    weight = weight, indices = input, offsets = offsets,
    scale_grad_by_freq = scale_grad_by_freq, mode = mode_enum,
    sparse = sparse, per_sample_weights = per_sample_weights,
    include_last_offset = include_last_offset
  )

  ret[[1]]
}

#' One_hot
#'
#' Takes LongTensor with index values of shape ``(*)`` and returns a tensor
#' of shape ``(*, num_classes)`` that have zeros everywhere except where the
#' index of last dimension matches the corresponding value of the input tensor,
#' in which case it will be 1.
#'
#' One-hot on Wikipedia: https://en.wikipedia.org/wiki/One-hot
#'
#'
#' @param tensor (LongTensor) class values of any shape.
#' @param num_classes (int) Total number of classes. If set to -1, the number
#'   of classes will be inferred as one greater than the largest class value in
#'   the input tensor.
#'
#' @export
nnf_one_hot <- function(tensor, num_classes = -1) {
  torch_one_hot(tensor, num_classes)
}
