Slice <- function(start = NULL, end = NULL, step = NULL) {
  cpp_torch_slice(cpp_optional_int64_t(start), cpp_optional_int64_t(end), 
                  cpp_optional_int64_t(step))
}

.torch_index <- function(x, slices) {
  
  x <- torch_randn(10, 10)
  slices <- c(0, TRUE)
  
  index <- cpp_torch_tensor_index_new()
  for (s in slices) {
    if (rlang::is_scalar_integerish(s))
      cpp_torch_tensor_index_append_int64(index, s)
    else if (is.na(el))
      cpp_torch_tensor_index_append_none(index)
    else if (is.logical(el))
      cpp_torch_tensor_index_append_bool(index, el)
  }
  
  p <- cpp_torch_tensor_index(x$ptr, index)
  Tensor$new(ptr = p)
}

slice_dim <- function(x, dim, s) {
  if (length(s) == 1 && all(is.na(s)))
    return(x)
  
  if (length(s) == 1)
    return(torch_select(x, dim = dim, index = ifelse(s > 0, s - 1, s)))
  
  if (inherits(s, "slice"))
    return(torch_slice(x, dim, s$start, s$end, s$step))
}

.d <- list(
  `:` = function(x, y) {
    
    if (inherits(x, "slice")) {
      x$step <- y
      return(x)
    }
    
    e <- list(start = ifelse(x > 0, x - 1, x), end = y, step = 1)
    attr(e, "class") <- "slice"
    e
  },
  
  N = .Machine$integer.max, 
  
  `..` = structure(list(), class = "fill")
)

#' @export
`[.torch_tensor` <- function(x, ...) {
  slices <- rlang::enquos(..., .ignore_empty = "none")
  
  slices <- lapply(slices, function(x) {
    if(rlang::is_missing(rlang::quo_get_expr(x))) 
      NA 
    else 
      lazyeval::lazy_eval(x, data = .d)
  })
  
  d <- dim(x)
  
  
  if (length(slices) <= length(d)) {
    if (inherits(slices[[1]], "fill")) {
      slices <- slices[-1]
      a <- as.list(rep(NA, length(d) - length(slices)))
      slices <- append(a, slices)
    } else if (inherits(slices[[length(slices)]], "fill")) {
      slices <- slices[-length(slices)]
      a <- as.list(rep(NA, length(d) - length(slices)))
      slices <- append(slices, a)
    }
  }
  
  if (length(slices) != length(d))
    stop("incorrect number of dimensions. Specified " , length(slices), " should be ",
         length(d), ".")
  
  index <- cpp_torch_tensor_index_new()
  for (el in slices) {
    if (rlang::is_scalar_integerish(el))
      cpp_torch_tensor_index_append_int64(index, el)
  }
  
  return(Tensor$new(ptr = cpp_torch_tensor_index(x$ptr, index)))
  
  for (dim in seq_along(slices)) {
    x <- slice_dim(x, dim - 1 - (length(d) - length(dim(x))), slices[[dim]])
  }
  x
}