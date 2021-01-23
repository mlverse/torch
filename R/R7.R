prepare_method <- function(m, active = FALSE) {
  
  if (active)
    attr(m, "active") <- TRUE
  
  m
}

#' @importFrom rlang env_bind
#' @importFrom rlang :=
R7Class <- function(classname = NULL, public = list(), private = list(),
                    active = list()) {
  
  methods <- new.env()
  private_methods <- new.env()
  
  public <- lapply(public, prepare_method)
  active <- lapply(active, prepare_method, active = TRUE)
  private <- lapply(private, prepare_method)
  
  env_bind(methods, !!!public)
  env_bind(methods, !!!active)
  env_bind(private_methods, !!!private)
  methods$private <- private_methods
  
  generator <- new.env(parent = methods)
  
  generator$new <- function(...) {
    self <- methods$initialize(...)
    class(self) <- c(classname, "R7")
    self
  }
  
  generator$set <- function(which, name, value) {
    if (which == "public")
      env_bind(methods, !!name := prepare_method(value))
    else if (which == "active")
      env_bind(methods, !!name := prepare_method(value, active = TRUE))
    else if (which == "private")
      env_bind(methods$private, !!name := prepare_method(value))
    else
      stop("can only set to public, private and active")
  }
  
  generator
}

extract_method <- function(self, name, call = TRUE) {
  
  if (inherits(self, "torch_tensor"))
    o <- mget(name, envir = Tensor, inherits = TRUE, ifnotfound = list(NULL))[[1]]
  else if (inherits(self, "torch_scalar"))
    o <- mget(name, envir = Scalar, inherits = TRUE, ifnotfound = list(NULL))[[1]]
  else if (inherits(self, "R7Private"))
    o <- mget(name, envir = unclass(self)$pvt, inherits = TRUE, ifnotfound = list(NULL))[[1]]
    
  if (name == "private") {
    o <- list(
      env = environment(),
      pvt = o
    )
    class(o) <- c("R7", "R7Private")
  }
    
  if (!is.function(o))
    return(o)
  
  if (!inherits(self, "R7Private"))
    private <- self$private

  if (inherits(self, "R7Private")) {
    environment(o) <- unclass(self)$env
  } else {
    environment(o) <- environment()  
  }
    
  if (call && isTRUE(attr(o, "active")))
    o()
  else
    o
}

#' @export
`$.R7` <- function(x, name) {
  extract_method(x, name)
}

#' @export
`$<-.R7` <- function(x, name, value) {
  f <- extract_method(x, name, call = FALSE)
  if (isTRUE(attr(f, "active"))) {
    f(value)
    invisible(x)
  } else {
    NextMethod("$<-", x)
  }
}

#' @export
`[[<-.R7` <- `$<-.R7`


#' @export
`[[.R7` <- `$.R7`

#' @export
print.R7 <- function(x, ...) {
  x$print(...)
}