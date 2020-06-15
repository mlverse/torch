Slice <- function(start = NULL, end = NULL, step = 1) {
  cpp_torch_slice(cpp_optional_int64_t(start), cpp_optional_int64_t(end), 
                  cpp_optional_int64_t(step))
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
  
  if (length(slices) == (length(d) + 1) && inherits(slices[[length(slices)]], "fill"))
    slices <- slices[1:(length(slices) -1)]
  
  if (length(slices) != length(d))
    stop("incorrect number of dimensions. Specified " , length(slices), " should be ",
         length(d), ".")
  
  index <- cpp_torch_tensor_index_new()
  for (s in slices) {
    if (rlang::is_scalar_integerish(s))
      cpp_torch_tensor_index_append_int64(index, ifelse(s > 0, s - 1, s))
    else if (rlang::is_scalar_atomic(s) && is.na(s))
      cpp_torch_tensor_index_append_slice(index, Slice())
    else if (is.logical(s))
      cpp_torch_tensor_index_append_bool(index, s)
    else if (inherits(s, "slice"))
      cpp_torch_tensor_index_append_slice(index, Slice(s$start, s$end, s$step))
    else if (rlang::is_integerish(s)) {
      
      if (all(s > 0)) {
        s <- s - 1
      } else if (all(s < 0)) {
        # nothing to do
      } else {
        value_error("All indices must be positive/or negative, not mixed.")
      }
      
      cpp_torch_tensor_index_append_tensor(index, torch_tensor(s, dtype = torch_long())$ptr)
    }
  }
  
  Tensor$new(ptr = cpp_torch_tensor_index(x$ptr, index))
}