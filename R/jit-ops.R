#' Enable idiomatic access to JIT operators from R.
#'
#' Call JIT operators directly from R, keeping the familiar argument types and argument order.
#' Note, however, that:
#' - all arguments are required (no defaults)
#' - axis numbering (as well as position numbers overall) starts from 0
#' - scalars have to be wrapped in `jit_scalar()`
#' 
#' @examples
#' t1 <- torch::torch_rand(4, 5)
#' t2 <- torch::torch_ones(5, 4)
#' # same as torch::torch_matmul(t1, t2)
#' jit_ops$aten$matmul(t1, t2)
#' 
#' # same as torch_split(torch::torch_arange(0, 3), 2, 1)
#' jit_ops$aten$split(torch::torch_arange(0, 3), torch::jit_scalar(2L), torch::jit_scalar(0L))
#' 

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


