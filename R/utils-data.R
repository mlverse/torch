Dataset <- R6::R6Class(
  classname = "dataset",
  lock_objects = FALSE,
  public = list(
    .getitem = function(index) {
      not_implemented_error()
    },
    state_dict = function() {
      # the default implementation will walk trough the public fields and try to
      # find tensors. It won't do it recursively, only flat fields will be 
      # considered.
      fields <- names(self)
      tensors <- list()
      for (f in fields) {
        value <- .subset2(self, f)
        if (inherits(value, "torch_tensor")) {
          tensors[[f]] <- value
        }
      }
      tensors
    },
    load_state_dict = function(x, ..., .refer_to_state_dict = FALSE) {
      # specially when using torch_load, it's possible to optimize by using the
      # .refer_to_state_dict field, so you don't need an extra copy.
      # you are not required to implement it, though, but add `...` to the
      # signature.
      if (.refer_to_state_dict) {
        for (nm in names(x)) {
          assign(nm, x[[nm]], envir = self)
        }
        invisible(NULL)
      } else {
        cli_abort("Loading the state_dict is only implemented when {.arg .refer_to_state_dict} is {.val TRUE}")
      }
    }
  )
)

IterableDataset <- R6::R6Class(
  classname = "iterable_dataset",
  lock_objects = FALSE,
  public = list(
    .iter = function() {
      not_implemented_error()
    },
    .length = function() {
      NA_integer_
    }
  )
)

is_map_dataset <- function(x) {
  inherits(x, "dataset")
}

is_iterable_dataset <- function(x) {
  inherits(x, "iterable_dataset")
}

#' @export
as_iterator.iterable_dataset <- function(x) {
  x$.iter()
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
#' the dataloader. `.getbatch` must work for batches of size larger or equal to 1 and 
#' care must be taken so it doesn't drop the batch dimension when it's queried with 
#' a length 1 batch index - for instance by using `drop=FALSE`. `.getitem()` is expected
#' to not include the batch dimension as it's added by the datalaoder.
#' For more on this see the the `vignette("loading-data")`.
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


#' Creates an iterable dataset
#' 
#' @inheritParams dataset
#' @examples
#' ids <- iterable_dataset(
#'   name = "hello",
#'   initialize = function(n = 5) {
#'     self$n <- n
#'     self$i <- 0
#'   },
#'   .iter = function() {
#'     i <- 0
#'     function() {
#'       i <<- i + 1
#'       if (i > self$n) {
#'         coro::exhausted()
#'       } else {
#'         i
#'       }
#'     }
#'   }
#' )
#' coro::collect(ids()$.iter())
#' @export
iterable_dataset <- function(name, inherit = IterableDataset, ..., 
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
    constructor_class = "iterable_dataset_generator"
  )
}

#' @export
print.dataset_generator <- function(x, ...) {
  cli::cat_line("<dataset_generator>")
  print(attr(x, "Dataset"))
}

#' @export
print.iterable_dataset_generator <- function(x, ...) {
  cli::cat_line("<iterable_dataset_generator>")
  print(attr(x, "IterableDataset"))
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
`[[.dataset` <- function(x, y) {
  if (is.character(y)) return(NextMethod("[[", x))
  y <- as.integer(y)
  stopifnot(length(y) == 1L)
  x$.getitem(y)
}

#' @export
length.dataset <- function(x) {
  x$.length()
}

#' @export
length.iterable_dataset <- function(x) {
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
  .getitem = function(index, ..., drop=TRUE) {
    if (is.list(index)) {
      index <- unlist(index)
    }

    lapply(self$tensors, function(x) {
      x[index, .., drop=drop]
    })
  },
  .getbatch = function(index) {
    self$.getitem(index, drop=FALSE)
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
    classes <- class(dataset)
    classes_to_append <- classes[classes != "R6"]
    class(self) <- c(paste0(classes_to_append, "_subset"), class(self))
  },
  .getitem = function(idx) {
    return(self$dataset[self$indices[idx]])
  },
  .length = function() {
    return(length(self$indices))
  }
)
