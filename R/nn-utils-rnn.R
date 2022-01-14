PackedSequence <- R6::R6Class(
  classname = "PackedSequence",
  lock_objects = FALSE,
  public = list(
    initialize = function(ptr = NULL) {
      self$ptr <- ptr
    }
  ),
  active = list(
    data = function() {
      Tensor$new(ptr = cpp_nn_utils_PackedSequence_data(self$ptr))
    },
    batch_sizes = function() {
      Tensor$new(ptr = cpp_nn_utils_PackedSequence_batch_sizes(self$ptr))
    },
    sorted_indices = function() {
      Tensor$new(ptr = cpp_nn_utils_PackedSequence_sorted_indices(self$ptr))$add(1L, alpha = 1L)
    },
    unsorted_indices = function() {
      Tensor$new(ptr = cpp_nn_utils_PackedSequence_unsorted_indices(self$ptr))$add(1L, alpha = 1L)
    }
  )
)

new_packed_sequence <- function(data, batch_sizes, sorted_indices, unsorted_indices) {
  o <- cpp_nn_utils_PackedSequence_new(
    data$ptr,
    batch_sizes$ptr,
    sorted_indices$sub(1L, 1L)$ptr,
    unsorted_indices$sub(1L, 1L)$ptr
  )
  PackedSequence$new(ptr = o)
}

is_packed_sequence <- function(x) {
  inherits(x, "PackedSequence")
}

#' Packs a Tensor containing padded sequences of variable length.
#'
#' `input` can be of size `T x B x *` where `T` is the length of the
#' longest sequence (equal to `lengths[1]`), `B` is the batch size, and
#' `*` is any number of dimensions (including 0). If `batch_first` is
#' `TRUE`, `B x T x *` `input` is expected.
#'
#' For unsorted sequences, use `enforce_sorted = FALSE`. If `enforce_sorted` is
#' `TRUE`, the sequences should be sorted by length in a decreasing order, i.e.
#' `input[,1]` should be the longest sequence, and `input[,B]` the shortest
#' one. `enforce_sorted = TRUE` is only necessary for ONNX export.
#'
#' @note
#' This function accepts any input that has at least two dimensions. You
#' can apply it to pack the labels, and use the output of the RNN with
#' them to compute the loss directly. A Tensor can be retrieved from
#' a `PackedSequence` object by accessing its `.data` attribute.
#'
#' @param input (Tensor): padded batch of variable length sequences.
#' @param lengths (Tensor): list of sequences lengths of each batch element.
#' @param batch_first (bool, optional): if `TRUE`, the input is expected in `B x T x *`
#'   format.
#' @param enforce_sorted (bool, optional): if `TRUE`, the input is expected to
#' contain sequences sorted by length in a decreasing order. If
#' `FALSE`, the input will get sorted unconditionally. Default: `TRUE`.
#'
#' @return
#' a `PackedSequence` object
#'
#' @export
nn_utils_rnn_pack_padded_sequence <- function(input, lengths, batch_first = FALSE,
                                              enforce_sorted = TRUE) {
  if (!is_torch_tensor(lengths)) {
    lengths <- torch_tensor(lengths, dtype = torch_long())
  }

  PackedSequence$new(cpp_nn_utils_rnn_pack_padded_sequence(
    input$ptr,
    lengths$ptr,
    batch_first,
    enforce_sorted
  ))
}

#' Packs a list of variable length Tensors
#'
#' `sequences` should be a list of Tensors of size `L x *`, where `L` is
#' the length of a sequence and `*` is any number of trailing dimensions,
#' including zero.
#'
#' For unsorted sequences, use `enforce_sorted = FALSE`. If `enforce_sorted`
#' is `TRUE`, the sequences should be sorted in the order of decreasing length.
#' `enforce_sorted = TRUE` is only necessary for ONNX export.
#'
#' @examples
#' x <- torch_tensor(c(1, 2, 3), dtype = torch_long())
#' y <- torch_tensor(c(4, 5), dtype = torch_long())
#' z <- torch_tensor(c(6), dtype = torch_long())
#'
#' p <- nn_utils_rnn_pack_sequence(list(x, y, z))
#' @param sequences `(list[Tensor])`: A list of sequences of decreasing length.
#' @param enforce_sorted (bool, optional): if `TRUE`, checks that the input
#'   contains sequences sorted by length in a decreasing order. If
#'   `FALSE`, this condition is not checked. Default: `TRUE`.
#'
#' @return
#' a `PackedSequence` object
#'
#' @export
nn_utils_rnn_pack_sequence <- function(sequences, enforce_sorted = TRUE) {
  PackedSequence$new(
    ptr = cpp_nn_utils_pack_sequence(sequences, enforce_sorted)
  )
}

#' Pads a packed batch of variable length sequences.
#'
#' It is an inverse operation to [nn_utils_rnn_pack_padded_sequence()].
#'
#' The returned Tensor's data will be of size `T x B x *`, where `T` is the length
#' of the longest sequence and `B` is the batch size. If `batch_first` is `TRUE`,
#' the data will be transposed into `B x T x *` format.
#'
#' @examples
#' seq <- torch_tensor(rbind(c(1, 2, 0), c(3, 0, 0), c(4, 5, 6)))
#' lens <- c(2, 1, 3)
#' packed <- nn_utils_rnn_pack_padded_sequence(seq, lens,
#'   batch_first = TRUE,
#'   enforce_sorted = FALSE
#' )
#' packed
#' nn_utils_rnn_pad_packed_sequence(packed, batch_first = TRUE)
#' @note
#' `total_length` is useful to implement the
#' `pack sequence -> recurrent network -> unpack sequence` pattern in a
#' `nn_module` wrapped in `~torch.nn.DataParallel`.
#'
#' @param sequence (PackedSequence): batch to pad
#' @param batch_first (bool, optional): if ``True``, the output will be in ``B x T x *`
#'    format.
#' @param padding_value (float, optional): values for padded elements.
#' @param total_length (int, optional): if not `NULL`, the output will be padded to
#'    have length `total_length`. This method will throw `ValueError`
#'    if `total_length` is less than the max sequence length in
#'    `sequence`.
#'
#' @return
#' Tuple of Tensor containing the padded sequence, and a Tensor
#' containing the list of lengths of each sequence in the batch.
#' Batch elements will be re-ordered as they were ordered originally when
#' the batch was passed to [nn_utils_rnn_pack_padded_sequence()] or
#' [nn_utils_rnn_pack_sequence()].
#'
#' @export
nn_utils_rnn_pad_packed_sequence <- function(sequence, batch_first = FALSE,
                                             padding_value = 0, total_length = NULL) {
  cpp_nn_utils_pad_packed_sequence(
    sequence$ptr, batch_first, padding_value,
    total_length
  )
}

#' Pad a list of variable length Tensors with `padding_value`
#'
#' `pad_sequence` stacks a list of Tensors along a new dimension,
#' and pads them to equal length. For example, if the input is list of
#' sequences with size `L x *` and if batch_first is False, and `T x B x *`
#' otherwise.
#'
#' `B` is batch size. It is equal to the number of elements in ``sequences``.
#' `T` is length of the longest sequence.
#' `L` is length of the sequence.
#' `*` is any number of trailing dimensions, including none.
#'
#' @examples
#' a <- torch_ones(25, 300)
#' b <- torch_ones(22, 300)
#' c <- torch_ones(15, 300)
#' nn_utils_rnn_pad_sequence(list(a, b, c))$size()
#' @note
#' This function returns a Tensor of size `T x B x *` or `B x T x *`
#' where `T` is the length of the longest sequence. This function assumes
#' trailing dimensions and type of all the Tensors in sequences are same.
#'
#' @param sequences `(list[Tensor])`: list of variable length sequences.
#' @param batch_first (bool, optional): output will be in `B x T x *` if `TRUE`,
#' or in `T x B x *` otherwise
#' @param padding_value (float, optional): value for padded elements. Default: 0.
#'
#' @return
#' Tensor of size `T x B x *` if `batch_first` is `FALSE`.
#' Tensor of size `B x T x *` otherwise
#'
#' @export
nn_utils_rnn_pad_sequence <- function(sequences, batch_first = FALSE, padding_value = 0) {
  o <- cpp_nn_utils_pad_sequence(sequences, batch_first, padding_value)
  Tensor$new(ptr = o)
}
