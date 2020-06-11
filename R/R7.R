prepare_method <- function(m, active = FALSE) {
  if (!rlang::is_function(m))
    return(m)
  
  
  rlang::fn_fmls(m) <- c(rlang::pairlist2(self=, private=), rlang::fn_fmls(m))
  
  if (active)
    attr(m, "active") <- TRUE
  
  m
}

R7Class <- function(classname = NULL, public = list(), private = list(),
                    active = list()) {
  
  methods <- new.env()
  private_methods <- new.env()
  
  public <- lapply(public, prepare_method)
  active <- lapply(active, prepare_method, active = TRUE)
  private <- lapply(private, prepare_method)
  
  rlang::env_bind(methods, !!!public)
  rlang::env_bind(methods, !!!active)
  rlang::env_bind(private_methods, !!!private)
  
  class(private_methods) <- "R7"
  methods$private <- private_methods
  
  generator <- new.env(parent = methods)
  
  generator$new <- function(...) {
    self <- new.env(parent = generator)
    class(self) <- c(classname, "R7")
    methods$initialize(self, self$private, ...)
    self
  }
  
  generator$set <- function(which, name, value) {
    if (which == "public")
      rlang::env_bind(methods, !!name := prepare_method(value))
    else if (which == "active")
      rlang::env_bind(methods, !!name := prepare_method(value, active = TRUE))
    else if (which == "private")
      rlang::env_bind(methods$private, !!name := prepare_method(value))
    else
      stop("can only set to public, private and active")
  }
  
  generator
}

#' @export
`$.R7` <- function(x, name) {
  o <- rlang::env_get(x, name, default = NULL, inherit = TRUE)
  
  if (name == "private")
    attr(o, "self") <- x
  
  if (!rlang::is_function(o))
    return(o)
  
  if (!is.null(attr(x, "self"))) {
    x <- attr(x, "self")
  }
    
  f <- function(...) {
    o(x, x$private, ...)
  }
  
  if (isTRUE(attr(o, "active")))
    f()
  else
    f
}

#' @export
print.R7 <- function(x, ...) {
  x$print()
}