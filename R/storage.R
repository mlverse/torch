#' @include tensor.R
NULL

Tensor$set("public", "storage", function() {
  Storage$new(cpp_Tensor_storage(self$ptr))
})

Tensor$set("public", "has_storage", function() {
  Storage$new(cpp_Tensor_has_storage(self$ptr))
})

Storage <- R6::R6Class(
  "torch_storage",
  lock_objects = FALSE,
  public = list(
    initialize = function(ptr) {
      self$ptr <- ptr
    },
    data_ptr = function() {
      cpp_Storage_data_ptr(self$ptr)
    }
  )
)
