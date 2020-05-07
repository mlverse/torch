list_with_default <- function(out_size, defaults) {
  
  if (is.numeric(out_size) && length(out_size) == 1)
    return(out_size)
  
  if (length(defaults) >= length(out_size))
    value_error("Input dimension should be at least {lenght(out_size) + 1}.")
  
  defaults <- tail(defaults, length(out_size))
    
  sapply(
    seq_along(out_size),
    function(i) {
      o <- out_size[[i]]
      if (is.null(o))
        o <- defaults[[i]]
      return(o)
    }
  )
}