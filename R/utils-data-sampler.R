
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
      function() {
        i <<- i + 1
        
        if (i > n)
          stop_iteration_error()
        
        i
      }
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
    initialize = function(data_source, replacement=FALSE, num_samples=NULL, generator = NULL) {
      self$data_source <- data_source
      self$replacement <- replacement
      self$.num_samples <- num_samples
      self$generator <- generator
    },
    .iter = function() {
      n <- length(self$data_source)
      
      if (self$replacement) {
        rand_tensor <- torch_randint(low = 1, high = n + 1, size = self$num_samples,
                                     dtype = torch_long(), generator = self$generator)
      } else {
        rand_tensor <- torch_randperm(n)$add(1L, 1L)#, generator = self$generator)
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
      if (is.null(self$.num_samples))
        length(self$data_source)
      else
        self$.num_samples
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
      s <- self$sampler$.iter()
      function() {
        batch <- list()
        for (i in seq_len(self$batch_size)) {
          
          er <- FALSE
          
          tryCatch(
            obs <- s(),
            stop_iteration_error = function(err) {
              er <<- TRUE
            }
          )
          
          if (er)
            break
          
          if (obs == coro::exhausted())
            break
          
          batch <- append(batch, obs)
        }
        
        if (length(batch) == self$batch_size)
          return(batch)
        
        if (length(batch) > 0 && !self$drop_last)
          return(batch)
          
        stop_iteration_error()
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
)