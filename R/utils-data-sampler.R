
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

SequentialSampler <- R6::R6Class(
  "utils_sampler_sequential",
  lock_objects = FALSE,
  inherit = Sampler,
  public = list(
    initialize = function(data_source) {
      self$data_source <- data_source
    },
    .iter = function() {
      i <- 0
      n <- length(self$data_source)
      coro::as_iterator(seq_len(n))
    },
    .length = function() {
      length(self$data_source)
    }
  )
)

RandomSampler <- R6::R6Class(
  classname = "utils_sampler_random",
  lock_objects = FALSE,
  inherit = Sampler,
  public = list(
    initialize = function(data_source, replacement = FALSE, num_samples = NULL, generator = NULL) {
      self$data_source <- data_source
      self$replacement <- replacement
      self$.num_samples <- num_samples
      self$generator <- generator
    },
    .iter = function() {
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
      as_iterator(rand_tensor)
    },
    .length = function() {
      self$num_samples
    }
  ),
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

BatchSampler <- R6::R6Class(
  classname = "utils_sampler_batch",
  lock_objects = FALSE,
  inherit = Sampler,
  public = list(
    initialize = function(sampler, batch_size, drop_last) {
      self$sampler <- sampler
      self$batch_size <- batch_size
      self$drop_last <- drop_last
    },
    .iter = function() {
      coro::generator(function() {
        batch <- list()

        for (idx in self$sampler) {
          batch[[length(batch) + 1]] <- idx
          if (length(batch) == self$batch_size) {
            yield(batch)
            batch <- list()
          }
        }

        if (length(batch) > 0 && !self$drop_last) {
          yield(batch)
        }
      })()
    },
    .length = function() {
      if (self$drop_last) {
        length(self$sampler) %/% self$batch_size
      } else {
        (length(self$sampler) + self$batch_size - 1) %/% self$batch_size
      }
    }
  )
)

#' @export
as_iterator.utils_sampler <- function(x) {
  it <- x$.iter()
  function() {
    it()
  }
}
