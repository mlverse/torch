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
  
  newaxis = NULL,
  
  `..` = structure(list(), class = "fill")
)

#' @importFrom rlang quo_is_missing
resolve_slices <- function(...) {
  slices <- enquos(..., .ignore_empty = "none")

  lapply(slices, function(x) {
    if(quo_is_missing(x)) 
      NA 
    else 
      eval_tidy(x, data = .d)
  })
}

#' @importFrom rlang is_function is_missing quo_get_expr enquos eval_tidy is_scalar_atomic is_integerish is_scalar_integerish
#' @export
`[.torch_tensor` <- function(x, ..., drop = TRUE) {
  
  slices <- resolve_slices(...)
  
  d <- cpp_tensor_dim(x$ptr)
  if (length(slices) < length(d)) {

    if (!inherits(slices[[1]], "fill") &&
        !inherits(slices[[length(slices)]], "fill"))
      value_error("incorrect number of dimensions")
  }
  
  # shortcut for selecting single indexes in the first dimensions.
  # in this case it's more efficient to use torch_select.
  if (drop) {
    i <- 1
    while(i <= length(slices) && is_scalar_integerish(slices[[i]])) {
      x <- Tensor$new(
        ptr = cpp_torch_namespace_select_self_Tensor_dim_int64_t_index_int64_t(
          x$ptr, 0, ifelse(slices[[i]] > 0, slices[[i]] - 1, slices[[i]])))
      i <- i + 1
    }  
    
    if (i > 1)
      slices <- slices[-seq_len(i-1)]
    
    if (length(slices) == 0)
      return(x)
    
    all_na <- TRUE
    for (e in slices) {
      if (length(e) == 0 || !is.na(e)) {
        all_na <- FALSE
        break
      }
    }
    
    if (all_na)
      return(x)
  }
  
  index <- cpp_torch_tensor_index_new()
  for (s in slices) {
    if (is_scalar_integerish(s)) {
      cpp_torch_tensor_index_append_int64(index, ifelse(s > 0, s - 1, s))   
      
      if (!drop)
        cpp_torch_tensor_index_append_none(index)
      
    } 
    else if (inherits(s, "fill")) 
      cpp_torch_tensor_index_append_ellipsis(index)
    else if (is_scalar_atomic(s) && is.na(s))
      cpp_torch_tensor_index_append_slice(index, Slice())
    else if (is.logical(s))
      cpp_torch_tensor_index_append_bool(index, s)
    else if (inherits(s, "slice"))
      cpp_torch_tensor_index_append_slice(index, Slice(s$start, s$end, s$step))
    else if (is.null(s))
      cpp_torch_tensor_index_append_none(index)
    else if (is_integerish(s)) {
    
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