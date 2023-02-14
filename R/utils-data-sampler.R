
#' @export
length.utils_sampler <- function(x) {
  x$.length()
}

Sampler <- R6::R6Class(
  classname = "utils_sampler",
  lock_objects = FALSE,
  public = list(
    initialize = function(data_source) {

    },
    .iter = function() {
      not_implemented_error()
    }
  )
)

#' Creates a new Sampler
#' 
#' Samplers can be used with [dataloader()] when creating batches from a torch
#' [dataset()].
#' 
#' A sampler must implement the `.iter` and `.length()` methods.
#' - `initialize` takes in a `data_source`. In general this is a [dataset()].
#' - `.iter` returns a function that returns a dataset index everytime it's called.
#' - `.length` returns the maximum number of samples that can be retrieved from
#'  that sampler.
#' 
#' @param name (optional) name of the sampler
#' @param inherit (optional) you can inherit from other samplers to re-use
#'   some methods.
#' @param ... Pass any number of fields or methods. You should at least define
#'   the `initialize` and `step` methods. See the examples section.
#' @param private (optional) a list of private methods for the sampler
#' @param active (optional) a list of active methods for the sampler.
#' @param parent_env used to capture the right environment to define the class.
#'   The default is fine for most situations.
#'
#' @export
sampler <- function(name = NULL, inherit = Sampler, ...,
                    private = NULL, active = NULL,
                    parent_env = parent.frame()) {
  create_class(
    name = name,
    inherit = inherit,
    ...,
    private = private,
    active = active,
    parent_env = parent_env,
    attr_name = "Sampler",
    constructor_class = c(name, "torch_sampler")
  )
}

SequentialSampler <- sampler(
  name = "utils_sampler_sequential",
  initialize = function(data_source) {
    self$data_source <- data_source
  },
  .iter_batch = function(batch_size) {
    n <- length(self$data_source)
    as_batch_iterator(seq_len(n), batch_size)
  },
  .iter = function() {
    self$.iter_batch(1L)
  },
  .length = function() {
    length(self$data_source)
  }
)

RandomSampler <- sampler(
  name = "utils_sampler_random",
  initialize = function(data_source, replacement = FALSE, num_samples = NULL, generator = NULL) {
    self$data_source <- data_source
    self$replacement <- replacement
    self$.num_samples <- num_samples
    self$generator <- generator
  },
  .iter_batch = function(batch_size) {
    n <- length(self$data_source)
    
    if (self$replacement) {
      rand_tensor <- torch_randint(
        low = 1, high = n + 1, size = self$num_samples,
        dtype = torch_long(), generator = self$generator
      )
    } else {
      rand_tensor <- torch_randperm(n)$add(1L, 1L) # , generator = self$generator)
    }
    rand_tensor <- as_array(rand_tensor$to(dtype = torch_int()))
    as_batch_iterator(rand_tensor, batch_size)
  },
  .iter = function() {
    self$.iter_batch(1L)
  },
  .length = function() {
    self$num_samples
  },
  active = list(
    num_samples = function() {
      if (is.null(self$.num_samples)) {
        length(self$data_source)
      } else {
        self$.num_samples
      }
    }
  )
)

BatchSampler <- sampler(
  name = "utils_sampler_batch",
  initialize = function(sampler, batch_size, drop_last) {
    self$sampler <- sampler
    self$batch_size <- batch_size
    self$drop_last <- drop_last
  },
  .iter_batch_sampler = function() {
    samp <- self$sampler$.iter_batch(self$batch_size)
    function() {
      batch <- samp()
      if (is_exhausted(batch)) return(batch)
      if (length(batch) == self$batch_size) return(batch)
      if (length(batch) > 0 && !self$drop_last) return(batch)
      coro::exhausted()
    }
  },
  .iter_sampler = function() {
    samp <- self$sampler$.iter()
    function() {
      batch <- list()
      repeat {
        id <- samp()
        if (is_exhausted(id)) break
        batch[[length(batch) + 1]] <- id
        if (length(batch) == self$batch_size) return(batch)
      }
      if (length(batch) > 0 && !self$drop_last) {
        return(batch)
      }
      coro::exhausted()
    }
  },
  .iter = function() {
    if (!is.null(self$sampler$.iter_batch)) {
      self$.iter_batch_sampler()
    } else {
      self$.iter_sampler()
    }
  },
  .length = function() {
    if (self$drop_last) {
      length(self$sampler) %/% self$batch_size
    } else {
      (length(self$sampler) + self$batch_size - 1) %/% self$batch_size
    }
  }
)

#' @export
as_iterator.utils_sampler <- function(x) {
  it <- x$.iter()
  function() {
    it()
  }
}

as_iterator_f <- function(x) {
  n <- length(x)
  i <- 0L
  function() {
    if (i == n) {
      return(coro::exhausted())
    }
    i <<- i + 1L
    x[i]
  }
}

as_batch_iterator <- function(x, batch_size) {
  n <- length(x)
  i <- 1L
  batch_size <- as.integer(batch_size)
  function() {
    if (i > n) {
      return(coro::exhausted())
    }
    end <- if ((i + batch_size - 1) > n) n else (i + batch_size - 1)
    out <- x[seq(i,end)]
    i <<- i + batch_size
    out
  }
}
