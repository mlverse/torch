Dataset <- R6::R6Class(
  classname = "utils_dataset", 
  public = list(
    get_item = function(index) {
      not_implemented_error()
    },
    add = function(other) {
      not_implemented_error()
    }
  )
)

util_dataset <- function(..., name = NULL) {
  R6::R6Class(
    classname = name,
    public = list(
      ...
    )
  )
}

TensorDataset <- R6::R6Class(
  classname = "utils_tensor_dataset",
  public = list(
    initialize = function(...) {
      tensors <- list(...)
      lens <- sapply(tensors, function(x) x$shape[1])
      
      if (!length(unique(lens)))
        value_error("all tensors must have the same size in the first dimension.")
      
      self$tensors <- tensors
    },
    get_item = function(index) {
      lapply(self$tensors, function(x) {
        x[index, ..]
      })
    },
    length = function() {
      self$tensors[[0]]$shape[1]
    }
  )
)

util_dataset_tensor <- function(...) {
  TensorDataset$new(...)
}