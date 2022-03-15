Dataset <- R6::R6Class(
  classname = "dataset",
  lock_objects = FALSE,
  public = list(
    .getitem = function(index) {
      not_implemented_error()
    }
  )
)

is_map_dataset <- function(x) {
  inherits(x, "dataset")
}

get_init <- function(x) {
  if (!is.null(x$public_methods$initialize)) {
    return(x$public_methods$initialize)
  } else {
    return(get_init(x$get_inherit()))
  }
}


#' Helper function to create an function that generates R6 instances of class `dataset`
#'
#' All datasets that represent a map from keys to data samples should subclass this
#' class. All subclasses should overwrite the `.getitem()` method, which supports
#' fetching a data sample for a given key. Subclasses could also optionally
#' overwrite `.length()`, which is expected to return the size of the dataset
#' (e.g. number of samples) used by many sampler implementations
#' and the default options of [dataloader()].
#'
#' @returns
#' The output is a function `f` with class `dataset_generator`. Calling `f()`
#' creates a new instance of the R6 class `dataset`. The R6 class is stored in the
#' enclosing environment of `f` and can also be accessed through `f`s attribute
#' `Dataset`.
#'
#' @section Get a batch of observations:
#'
#' By default datasets are iterated by returning each observation/item individually.
#' Often it's possible to have an optimized implementation to take a batch
#' of observations (eg, subsetting a tensor by multiple indexes at once is faster than
#' subsetting once for each index), in this case you can implement a `.getbatch` method
#' that will be used instead of `.getitem` when getting a batch of observations within
#' the dataloader. `.getbatch` must work for batches of size larger or equal to 1. For more
#' on this see the the `vignette("loading-data")`.
#'
#' @note
#' [dataloader()]  by default constructs a index
#' sampler that yields integral indices.  To make it work with a map-style
#' dataset with non-integral indices/keys, a custom sampler must be provided.
#'
#' @param name a name for the dataset. It it's also used as the class
#'   for it.
#' @param inherit you can optionally inherit from a dataset when creating a
#'   new dataset.
#' @param ... public methods for the dataset class
#' @param parent_env An environment to use as the parent of newly-created
#'   objects.
#' @inheritParams nn_module
#'
#' @export
dataset <- function(name = NULL, inherit = Dataset, ...,
                    private = NULL, active = NULL,
                    parent_env = parent.frame()) {
  create_class(
    name = name,
    inherit = inherit,
    ...,
    private = private,
    active = active,
    parent_env = parent_env,
    attr_name = "Dataset",
    constructor_class = "dataset_generator"
  )
}

#' @export
print.dataset_generator <- function(x, ...) {
  cli::cat_line("<dataset_generator>")
  print(attr(x, "Dataset"))
}

#' @export
`[.dataset` <- function(x, y) {
  y <- as.integer(y)
  if (!is.null(x$.getbatch)) {
    x$.getbatch(y)
  } else {
    x$.getitem(y)
  }
}

#' @export
length.dataset <- function(x) {
  x$.length()
}

#' Dataset wrapping tensors.
#'
#' Each sample will be retrieved by indexing tensors along the first dimension.
#'
#' @param ... tensors that have the same size of the first dimension.
#'
#' @export
tensor_dataset <- dataset(
  name = "tensor_dataset",
  initialize = function(...) {
    tensors <- rlang::list2(...)
    lens <- sapply(tensors, function(x) x$shape[1])

    if (!length(unique(lens))) {
      value_error("all tensors must have the same size in the first dimension.")
    }

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
  .getbatch = function(index) {
    self$.getitem(index)
  },
  .length = function() {
    self$tensors[[1]]$shape[1]
  }
)

#' Dataset Subset
#'
#' Subset of a dataset at specified indices.
#'
#' @param dataset  (Dataset): The whole Dataset
#' @param indices  (sequence): Indices in the whole set selected for subset
#'
#' @export
dataset_subset <- dataset(
  initialize = function(dataset, indices) {
    self$dataset <- dataset
    self$indices <- indices
    if (!is.null(dataset$.getbatch)) {
      self$.getbatch <- self$.getitem
    }
  },
  .getitem = function(idx) {
    return(self$dataset[self$indices[idx]])
  },
  .length = function() {
    return(length(self$indices))
  }
)
