list_with_default <- function(out_size, defaults) {
  if (is.numeric(out_size) && length(out_size) == 1) {
    return(out_size)
  }

  if (length(defaults) >= length(out_size)) {
    value_error("Input dimension should be at least {lenght(out_size) + 1}.")
  }

  defaults <- utils::tail(defaults, length(out_size))

  sapply(
    seq_along(out_size),
    function(i) {
      o <- out_size[[i]]
      if (is.null(o)) {
        o <- defaults[[i]]
      }
      return(o)
    }
  )
}

nn_util_ntuple <- function(n) {
  function(x) {
    if (length(x) > 1) {
      return(x)
    }

    rep(x, n)
  }
}

nn_util_single <- nn_util_ntuple(1)
nn_util_pair <- nn_util_ntuple(2)
nn_util_triple <- nn_util_ntuple(3)

nn_util_reverse_repeat_tuple <- function(t, n) {
  rep(rev(t), each = n)
}
