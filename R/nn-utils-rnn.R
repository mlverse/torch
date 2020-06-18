
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

nn_utils_rnn_pack_padded_sequences <- function(input, lengths, batch_first=FALSE, 
                                               enforce_sorted=TRUE) {
  
  if (!is_torch_tensor(lengths))
    lengths <- torch_tensor(lengths, dtype=torch_long())
  
  PackedSequence$new(cpp_nn_utils_rnn_pack_padded_sequence(input$ptr,
                                                           lengths$ptr,
                                                           batch_first,
                                                           enforce_sorted))
}

