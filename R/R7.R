prepare_method <- function(m) {
  if (!rlang::is_function(m))
    return(m)
  
  
  rlang::fn_fmls(m) <- c(rlang::pairlist2(self=), rlang::fn_fmls(m))
  m
}

R7Class <- function(classname = NULL, public = list(), private = NULL,
                    active = list()) {
  
  methods <- new.env()
  
  public <- lapply(public, prepare_method)
  active <- lapply(active, prepare_method)
  
  rlang::env_bind(methods, !!!public)
  rlang::env_bind_active(methods, !!!active)
  
  generator <- new.env(parent = methods)
  
  generator$new <- function(...) {
    self <- new.env(parent = generator)
    class(self) <- c(classname, "R7")
    methods$initialize(self, ...)
    self
  }
  
  generator$set <- function(which, name, value) {
    if (which == "public" || which == "private")
      rlang::env_bind(methods, !!name := prepare_method(value))
    else if (which == "active")
      rlang::env_bind_active(methods, !!name := prepare_method(value))
    else
      stop("can only set to public and active")
  }
  
  generator
}

`$.R7` <- function(x, name) {
  o <- rlang::env_get(x, name, default = NULL, inherit = TRUE)
  
  if (!rlang::is_function(o))
    return(o)
  
  function(...) {
    o(x, ...)
  }
}