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

#' Embedding bag module
#'
#' Computes sums, means or maxes of `bags` of embeddings, without instantiating the
#' intermediate embeddings.
#' 
#' @param num_embeddings (int): size of the dictionary of embeddings
#' @param embedding_dim (int): the size of each embedding vector
#' @param max_norm (float, optional): If given, each embedding vector with norm larger than `max_norm`
#'   is renormalized to have norm `max_norm`.
#' @param norm_type (float, optional): The p of the p-norm to compute for the `max_norm` option. Default ``2``
#' @param scale_grad_by_freq (boolean, optional): If given, this will scale gradients by the inverse of frequency of
#'   the words in the mini-batch. Default ``False``.
#' @param mode (string, optional): ``"sum"``, ``"mean"`` or ``"max"``. Specifies the way to reduce the bag.
#' ``"sum"`` computes the weighted sum, taking `per_sample_weights`  into consideration. ``"mean"`` computes 
#' the average of the values in the bag, ``"max"`` computes the max value over each bag.
#' @param sparse (bool, optional): If ``True``, gradient w.r.t. `weight` matrix will be a sparse tensor.
#' See Notes for more details regarding sparse gradients.
#' @param include_last_offset (bool, optional): if ``True``, `offsets` has one additional element, where the last element
#' is equivalent to the size of `indices`. This matches the CSR format.
#' @param padding_idx (int, optional):  If given, pads the output with the embedding vector at `padding_idx`
#'   (initialized to zeros) whenever it encounters the index.
#' @param .weight (Tensor, optional) embeddings weights (in case you want to set it manually)
#'
#' @section Attributes:
#' - weight (Tensor): the learnable weights of the module of shape (num_embeddings, embedding_dim)
#'   initialized from \eqn{\mathcal{N}(0, 1)}
#'
#' @examples
#' # an EmbeddingBag module containing 10 tensors of size 3
#' embedding_sum <- nn_embedding_bag(10, 3, mode = 'sum')
#' # a batch of 2 samples of 4 indices each
#' input <- torch_tensor(c(1, 2, 4, 5, 4, 3, 2, 9), dtype = torch_long())
#' offsets <- torch_tensor(c(0, 4), dtype = torch_long())
#' embedding_sum(input, offsets)
#' # example with padding_idx
#' embedding_sum <- nn_embedding_bag(10, 3, mode = 'sum', padding_idx = 1)
#' input <- torch_tensor(c(2, 2, 2, 2, 4, 3, 2, 9), dtype = torch_long())
#' offsets <- torch_tensor(c(0, 4), dtype = torch_long())
#' embedding_sum(input, offsets)
#' # An EmbeddingBag can be loaded from an Embedding like so
#' embedding <- nn_embedding(10, 3, padding_idx = 2)
#' embedding_sum <- nn_embedding_bag$from_pretrained(embedding$weight,
#'                                                  padding_idx = embedding$padding_idx,
#'                                                  mode='sum')
#' @export
nn_embedding_bag <- nn_module(
  "nn_embedding_bag",
  initialize = function(num_embeddings, embedding_dim, max_norm = NULL, 
                        norm_type = 2, scale_grad_by_freq = FALSE, 
                        mode = "mean", sparse = FALSE, 
                        include_last_offset = FALSE, padding_idx = NULL,
                        .weight = NULL) {
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
    
    self$mode = mode
    self$sparse = sparse
    self$include_last_offset = include_last_offset
    
    },
  reset_parameters = function() {
    nn_init_normal_(self$weight)
    if (!is.null(self$padding_idx)) {
      with_no_grad({
        self$weight[self$padding_idx, ..]$fill_(0)
      })
    }
  },
  forward = function(input, offsets = NULL, per_sample_weights = NULL) {
    nnf_embedding_bag(
      input, self$weight, offsets, self$max_norm, self$norm_type, 
      self$scale_grad_by_freq, self$mode, self$sparse, per_sample_weights,
      self$include_last_offset, self$padding_idx
    )
  }
) 

