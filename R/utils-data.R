Dataset <- R6::R6Class(
  classname = "utils_dataset", 
  lock_objects = FALSE,
  public = list(
    get_item = function(index) {
      not_implemented_error()
    },
    add = function(other) {
      not_implemented_error()
    }
  )
)

is_map_dataset <- function(x) {
  inherits(x, "utils_dataset")
}

#' An abstract class representing a `Dataset`.
#' 
#' All datasets that represent a map from keys to data samples should subclass
#' it. All subclasses should overwrite `get_item`, supporting fetching a
#' data sample for a given key. Subclasses could also optionally overwrite
#' `lenght`, which is expected to return the size of the dataset by many
#' `~torch.utils.data.Sampler` implementations and the default options
#' of `~torch.utils.data.DataLoader`.
#' 
#' @note 
#' `~torch.utils.data.DataLoader` by default constructs a index
#' sampler that yields integral indices.  To make it work with a map-style
#' dataset with non-integral indices/keys, a custom sampler must be provided.
#' 
#' @param ... public methods for the dataset class
#' 
#' @export
utils_dataset <- function(..., name = NULL) {
  R6::R6Class(
    classname = name,
    lock_objects = FALSE,
    inherit = Dataset,
    public = list(
      ...
    )
  )
}

#' @export
`[.utils_dataset` <- function(x, y) {
  x$.getitem(y)
}

#' @export
length.utils_dataset <- function(x) {
  x$.length()
}

TensorDataset <- utils_dataset(
  name = "utils_dataset_tensor",
  initialize = function(...) {
    tensors <- list(...)
    lens <- sapply(tensors, function(x) x$shape[1])
    
    if (!length(unique(lens)))
      value_error("all tensors must have the same size in the first dimension.")
    
    self$tensors <- tensors
  },
  .getitem = function(index) {
    
    if (is.list(index)) {
      index <- unlist(index)
    }
    
    lapply(self$tensors, function(x) {
        x[index, ..]
    })
  },
  .length = function() {
    self$tensors[[1]]$shape[1]
  }
)

#' Dataset wrapping tensors.
#' 
#' Each sample will be retrieved by indexing tensors along the first dimension.
#' 
#' @param ... tensors that have the same size of the first dimension.
#'
#' @export
utils_dataset_tensor <- function(...) {
  TensorDataset$new(...)
}

