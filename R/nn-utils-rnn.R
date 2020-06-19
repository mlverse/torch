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
      Tensor$new(ptr = cpp_nn_utils_PackedSequence_sorted_indices(self$ptr))
    },
    unsorted_indices = function() {
      Tensor$new(ptr = cpp_nn_utils_PackedSequence_unsorted_indices(self$ptr))
    }
  )
)

nn_utils_rnn_pack_padded_sequence <- function(input, lengths, batch_first=FALSE, 
                                               enforce_sorted=TRUE) {
  
  if (!is_torch_tensor(lengths))
    lengths <- torch_tensor(lengths, dtype=torch_long())
  
  PackedSequence$new(cpp_nn_utils_rnn_pack_padded_sequence(input$ptr,
                                                           lengths$ptr,
                                                           batch_first,
                                                           enforce_sorted))
}

nn_utils_rnn_pack_sequence <- function(sequences, enforce_sorted = TRUE) {
  
  if (is_torch_tensor(sequences))
    sequences <- list(sequences)
  
  sequences <- TensorList$new(x = sequences)
  PackedSequence$new(
    ptr = cpp_nn_utils_pack_sequence(sequences$ptr, enforce_sorted)
  )
}

nn_utils_rnn_pad_padded_sequence <- function(sequence, batch_first = FALSE, 
                                             padding_value = 0, total_lenght = NULL) {
  o <- cpp_nn_utils_pad_packed_sequence(sequence$ptr, batch_forst, padding_value, 
                                        cpp_optional_int64_t(total_lenght))
  TensorList$new(ptr = o)$to_r()
}

nn_utils_rnn_pad_sequence <- function(sequence, batch_first = FALSE, padding_value = 0) {
  o <- cpp_nn_utils_pad_sequence(sequence$ptr, batch_first, padding_value)
  Tensor$new(ptr = o)
}

