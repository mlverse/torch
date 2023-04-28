#' @export
jit_ops <- structure(list(), class = "torch_ops")

#' @export
.DollarNames.torch_ops <- function(x, pattern = "") {
  candidates <- cpp_jit_all_operators()
  if (length(x) == 0) {
    unique(sub("::(.)*", "", candidates))
  } else if (length(x) == 1) {
    namespace <- x[[1]]
    candidates <- unique(grep(paste0(namespace, "::"), candidates, value = T))
    sub(paste0(namespace, "::"), "", candidates)
  }
}

#' @export
`$.torch_ops` <- function(x, y, ...) {
  if (length(x) == 0) {
    return (structure(list(y), class = "torch_ops"))
  }
  op <- function(...) {
    lst <- cpp_jit_execute(paste(x[[1]], y, sep = "::"), list(...))
    if (length(lst) == 1) lst[[1]] else lst 
  }
  class(op) <- "torch_ops"
  attr(op, "opname") <- paste0(x[[1]], "::", y)
  op
}

#' @export
print.torch_ops <- function(x, ...) {
  if (length(x) == 1) {
    if (typeof(x) == "closure") {
      opname <- attr(x, "opname")
      info <- cpp_jit_operator_info(opname)
      print(info)
    } else if ((typeof(x) == "list") && typeof(x[[1]]) == "character") {
      cat("<torch_ops>: Handle to namespace ", x[[1]], "\n")
    }
  } else {
    cat("Object of class <torch_ops>\n")
  }
}


