as_iterator <- function(x) {
  i <- 0
  n <- length(x)
  function() {
    i <<- i + 1
    
    if (i > n)
      return(NULL)
    
    x[[i]]
  }
}

Sampler <- R6::R6Class(
  classname = "utils_sampler",
  lock_objects = FALSE,
  public = list(
    initialize = function(data_source) {
      
    },
    iter = function() {
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
    iter = function() {
      i <- 0
      n <- length(self$data_source)
      function() {
        i <<- i + 1
        
        if (i > n)
          return(NULL)
        
        i
      }
    },
    lenght = function() {
      lenght(self$data_source)
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
      self$num_samples_ <- num_samples
      self$generator <- generator
    },
    iter = function() {
      n <- length(self$data_source)
      if (self$replacement) {
        rand_tensor <- torch_randint(low = 1, high = n, size = self$num_samples,
                                     dtype = torch_long(), generator = self$generator)
      } else {
        rand_tensor <- torch_randperm(n, generaor = self$generator)
      }
      rand_tensor <- as_array(rand_tensor)
      as_iterator(rand_tensor)
    },
    length = function() {
      self$num_samples
    }
  ),
  active = list(
    num_samples = function() {
      if (is.null(self$num_samples_))
        length(self$data_source)
      else
        self$num_samples_
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
    iter = function() {
      s <- sampler$iter()
      function() {
        batch <- list()
        for (i in seq_len(self$batch_size)) {
          i <- s()
          
          if (is.null(i))
            break
          
          batch <- append(batch, i)
        }
        
        if (length(batch) == self$batch_size)
          return(batch)
        
        if (length(batch) > 0 && !self$drop_last)
          return(batch)
          
        NULL
      }
    },
    length = function() {
      if (self$drop_last) {
        lenght(self$sampler) %/% self$batch_size
      } else {
        (lenght(self$sampler) + self$batch_size - 1) %/% self$batch_size
      }
    }
  )
)