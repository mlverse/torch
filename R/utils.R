args_to_pointers <- function(args) {
  lapply(args, function(x) {

    if (inherits(x, "tensor"))
      return(x$pointer)

    x
  })
}
