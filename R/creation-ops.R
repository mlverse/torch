resolve_size <- function(...) {
  size <- list(...)
  
  if (!is.null(size[["size"]])) {
    
    if (!length(size) == 1)
      stop("You should specify a single size argument.")
    
    size <- size[["size"]]
  } else if (length(size[[1]]) > 1) {
    
    if (!length(size) == 1)
      stop("You should specify a single size argument.")
    
    size <- size[[1]]
  }
  
  size
}

torch_ones <- function(..., names = NULL, dtype = NULL, layout = torch_strided(), 
  device=NULL, requires_grad = FALSE) {
  args <- list(
    options = list(
      dtype = dtype,
      layout = layout,
      device = device,
      requires_grad = requires_grad
    )
  )
  args$size <- resolve_size(...)
  if (!is.null(names)) args$names <- names
  do.call(.torch_ones, args)
}

torch_ones_like <- function(input, dtype = NULL, layout = torch_strided(), 
                            device=NULL, requires_grad = FALSE, 
                            memory_format = torch_preserve_format()) {
  args <- list(
    options = list(
      dtype = dtype,
      layout = layout,
      device = device,
      requires_grad = requires_grad
    )
  )
  args$memory_format <- memory_format
  args$self <- input
  do.call(.torch_ones_like, args)
}

torch_rand <- function(..., names = NULL, dtype = NULL, layout = torch_strided(), 
                       device=NULL, requires_grad = FALSE) {
  args <- list(
    options = list(
      dtype = dtype,
      layout = layout,
      device = device,
      requires_grad = requires_grad
    )
  )
  args$size <- resolve_size(...)
  if (!is.null(names)) args$names <- names
  do.call(.torch_rand, args)
}

torch_rand_like <- function(input, dtype = NULL, layout = torch_strided(), 
                            device=NULL, requires_grad = FALSE, 
                            memory_format = torch_preserve_format()) {
  args <- list(
    options = list(
      dtype = dtype,
      layout = layout,
      device = device,
      requires_grad = requires_grad
    )
  )
  args$memory_format <- memory_format
  args$self <- input
  do.call(.torch_rand_like, args)
}

torch_randint <- function(low, high, size, generator = NULL, dtype = NULL, layout = torch_strided(), 
                          device=NULL, requires_grad = FALSE, 
                          memory_format = torch_preserve_format()) {
  
  args <- list(
    options = list(
      dtype = dtype,
      layout = layout,
      device = device,
      requires_grad = requires_grad
    )
  )
  args$low <- low
  args$high <- high
  args$size <- size
  if (!is.null(generator)) args$generator <- NULL
  
  do.call(.torch_randint, args)
}

torch_randint_like <- function(input, low, high, dtype = NULL, 
                               layout = torch_strided(), 
                               device=NULL, requires_grad = FALSE) {
  args <- list(
    options = list(
      dtype = dtype,
      layout = layout,
      device = device,
      requires_grad = requires_grad
    )
  )
  args$self <- input
  args$low <- low
  args$high <- high
  
  do.call(.torch_randint_like, args)
}

torch_randn <- function(..., names = NULL, dtype = NULL, layout = torch_strided(), 
                       device=NULL, requires_grad = FALSE) {
  args <- list(
    options = list(
      dtype = dtype,
      layout = layout,
      device = device,
      requires_grad = requires_grad
    )
  )
  args$size <- resolve_size(...)
  if (!is.null(names)) args$names <- names
  do.call(.torch_randn, args)
}

torch_randn_like <- function(input, dtype = NULL, layout = torch_strided(), 
                            device=NULL, requires_grad = FALSE, 
                            memory_format = torch_preserve_format()) {
  args <- list(
    options = list(
      dtype = dtype,
      layout = layout,
      device = device,
      requires_grad = requires_grad
    )
  )
  args$memory_format <- memory_format
  args$self <- input
  do.call(.torch_randn_like, args)
}

torch_randperm <- function(n, dtype = torch_int64(), layout = torch_strided(), 
                           device=NULL, requires_grad = FALSE) {
  args <- list(
    options = list(
      dtype = dtype,
      layout = layout,
      device = device,
      requires_grad = requires_grad
    )
  )
  args$n <- n
  do.call(.torch_randperm, args)
}

torch_zeros <- function(..., names = NULL, dtype = NULL, layout = torch_strided(), 
                       device=NULL, requires_grad = FALSE) {
  args <- list(
    options = list(
      dtype = dtype,
      layout = layout,
      device = device,
      requires_grad = requires_grad
    )
  )
  args$size <- resolve_size(...)
  if (!is.null(names)) args$names <- names
  do.call(.torch_zeros, args)
}

torch_zeros_like <- function(input, dtype = NULL, layout = torch_strided(), 
                            device=NULL, requires_grad = FALSE, 
                            memory_format = torch_preserve_format()) {
  args <- list(
    options = list(
      dtype = dtype,
      layout = layout,
      device = device,
      requires_grad = requires_grad
    )
  )
  args$memory_format <- memory_format
  args$self <- input
  do.call(.torch_zeros_like, args)
}

torch_empty <- function(..., names = NULL, dtype = NULL, layout = torch_strided(), 
                        device=NULL, requires_grad = FALSE) {
  args <- list(
    options = list(
      dtype = dtype,
      layout = layout,
      device = device,
      requires_grad = requires_grad
    )
  )
  args$size <- resolve_size(...)
  if (!is.null(names)) args$names <- names
  do.call(.torch_empty, args)
}

torch_empty_like <- function(input, dtype = NULL, layout = torch_strided(), 
                             device=NULL, requires_grad = FALSE, 
                             memory_format = torch_preserve_format()) {
  args <- list(
    options = list(
      dtype = dtype,
      layout = layout,
      device = device,
      requires_grad = requires_grad
    )
  )
  args$memory_format <- memory_format
  args$self <- input
  do.call(.torch_empty_like, args)
}

# "arange", "range", "linspace",
# "logspace", "eye", "empty", "empty_like", "empty_strided",
# "full", "full_like"