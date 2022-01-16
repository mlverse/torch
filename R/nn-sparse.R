#' @include nn.R
NULL

#' Embedding module
#'
#' A simple lookup table that stores embeddings of a fixed dictionary and size.
#' This module is often used to store word embeddings and retrieve them using indices.
#' The input to the module is a list of indices, and the output is the corresponding
#' word embeddings.
#'
#' @param num_embeddings (int): size of the dictionary of embeddings
#' @param embedding_dim (int): the size of each embedding vector
#' @param padding_idx (int, optional): If given, pads the output with the embedding vector at `padding_idx`
#'   (initialized to zeros) whenever it encounters the index.
#' @param max_norm (float, optional): If given, each embedding vector with norm larger than `max_norm`
#'   is renormalized to have norm `max_norm`.
#' @param norm_type (float, optional): The p of the p-norm to compute for the `max_norm` option. Default ``2``.
#' @param scale_grad_by_freq (boolean, optional): If given, this will scale gradients by the inverse of frequency of
#'   the words in the mini-batch. Default ``False``.
#' @param sparse (bool, optional): If ``True``, gradient w.r.t. `weight` matrix will be a sparse tensor.
#' @param .weight (Tensor) embeddings weights (in case you want to set it manually)
#'
#' See Notes for more details regarding sparse gradients.
#'
#' @section Attributes:
#' - weight (Tensor): the learnable weights of the module of shape (num_embeddings, embedding_dim)
#'   initialized from \eqn{\mathcal{N}(0, 1)}
#'
#' @section Shape:
#' - Input: \eqn{(*)}, LongTensor of arbitrary shape containing the indices to extract
#' - Output: \eqn{(*, H)}, where `*` is the input shape and \eqn{H=\mbox{embedding\_dim}}
#'
#' @note
#' Keep in mind that only a limited number of optimizers support
#' sparse gradients: currently it's `optim.SGD` (`CUDA` and `CPU`),
#' `optim.SparseAdam` (`CUDA` and `CPU`) and `optim.Adagrad` (`CPU`)
#'
#' @note
#' With `padding_idx` set, the embedding vector at
#' `padding_idx` is initialized to all zeros. However, note that this
#' vector can be modified afterwards, e.g., using a customized
#' initialization method, and thus changing the vector used to pad the
#' output. The gradient for this vector from [nn_embedding]
#' is always zero.
#'
#' @examples
#' # an Embedding module containing 10 tensors of size 3
#' embedding <- nn_embedding(10, 3)
#' # a batch of 2 samples of 4 indices each
#' input <- torch_tensor(rbind(c(1, 2, 4, 5), c(4, 3, 2, 9)), dtype = torch_long())
#' embedding(input)
#' # example with padding_idx
#' embedding <- nn_embedding(10, 3, padding_idx = 1)
#' input <- torch_tensor(matrix(c(1, 3, 1, 6), nrow = 1), dtype = torch_long())
#' embedding(input)
#' @export
nn_embedding <- nn_module(
  "nn_embedding",
  initialize = function(num_embeddings, embedding_dim, padding_idx = NULL,
                        max_norm = NULL, norm_type = 2, scale_grad_by_freq = FALSE,
                        sparse = FALSE, .weight = NULL) {
    self$num_embeddings <- num_embeddings
    self$embedding_dim <- embedding_dim

    if (!is.null(padding_idx)) {
      if (padding_idx > 0 && padding_idx > num_embeddings) {
        value_error("padding idx must be within num_embeddings")
      } else if (padding_idx < 0) {
        if (padding_idx <= (-num_embeddings)) {
          value_error("padding idx must be within num_embeddings")
        }

        padding_idx <- self$num_embeddings + padding_idx
      }
    }

    self$padding_idx <- padding_idx
    self$max_norm <- max_norm
    self$norm_type <- norm_type
    self$scale_grad_by_freq <- scale_grad_by_freq

    if (is.null(.weight)) {
      self$weight <- nn_parameter(torch_empty(num_embeddings, embedding_dim))
      self$reset_parameters()
    } else {
      if (!all(.weight$size() == c(num_embeddings, embedding_dim))) {
        value_error("Shape of weight does not match num_embeddings and embedding_dim")
      }

      self$weight <- nn_parameter(.weight)
    }

    self$sparse <- sparse
  },
  reset_parameters = function() {
    nn_init_normal_(self$weight)
    if (!is.null(self$padding_idx)) {
      with_no_grad({
        self$weight[self$padding_idx, ..]$fill_(0)
      })
    }
  },
  forward = function(input) {
    nnf_embedding(
      input, self$weight, self$padding_idx, self$max_norm,
      self$norm_type, self$scale_grad_by_freq, self$sparse
    )
  }
)
