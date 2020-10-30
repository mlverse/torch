Dataset <- R6::R6Class(
  classname = "dataset", 
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
  inherits(x, "dataset")
}

get_init <- function(x) {
  
  if (!is.null(x$public_methods$initialize))
    return(x$public_methods$initialize)
  else
    return(get_init(x$get_inherit()))
  
}

#' Helper function to create an R6 class that inherits from the abstract `Dataset` class
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
#' @param name a name for the dataset. It it's also used as the class
#'   for it.
#' @param inherit you can optionally inherit from a dataset when creating a 
#'   new dataset.
#' @param ... public methods for the dataset class
#' @param parent_env An environment to use as the parent of newly-created 
#'   objects.
#' 
#' @export
dataset <- function(name = NULL, inherit = Dataset, ..., parent_env = parent.frame()) {
  
  args <- list(...)
  
  active <- args$active
  if (!is.null(active))
    args <- args[-which(names(args) == "active")]
  
  if (!is.null(attr(inherit, "Dataset")))
    inherit <- attr(inherit, "Dataset")
  
  e <- new.env(parent = parent_env)
  e$inherit <- inherit
    
  d <- R6::R6Class(
    classname = name,
    lock_objects = FALSE,
    inherit = inherit,
    public = args,
    active = active,
    parent_env = e
  )
  
  init <- get_init(d)
  # same signature as the init method, but calls with dataset$new.
  f <- rlang::new_function(
    args = rlang::fn_fmls(init),
    body = rlang::expr({
      d$new(!!!rlang::fn_fmls_syms(init))
    })
  )

  attr(f, "Dataset") <- d
  f
}

#' @export
`[.dataset` <- function(x, y) {
  x$.getitem(y)
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







