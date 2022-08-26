prepare_method <- function(m, active = FALSE) {
  if (active) {
    attr(m, "active") <- TRUE
  }

  m
}

.generators <- new.env(parent = emptyenv())
.r7_env <- new.env()

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
    if (which == "public") {
      env_bind(methods, !!name := prepare_method(value))
    } else if (which == "active") {
      env_bind(methods, !!name := prepare_method(value, active = TRUE))
    } else if (which == "private") {
      env_bind(methods$private, !!name := prepare_method(value))
    } else {
      stop("can only set to public, private and active")
    }
  }

  # set the generator/classname env
  .generators[[classname]] <- generator

  generator
}

#' @importFrom rlang env_get
extract_method <- function(self, name, call = TRUE) {

  # resolve the class environment
  # every R7 class is registered in the .generators env
  # so we can find the generator give the class name
  is_private <- inherits(self, "R7Private")
  if (is_private) {
    e <- unclass(self)$pvt
  } else {
    e <- .generators[[class(self)[1]]]
  }

  # gets the object from the class environment
  o <- env_get(
    env = e, 
    nm = name, 
    inherit = TRUE,
    default = NULL
  )

  if (name == "private") {
    o <- structure(
      list(
        env = environment(),
        pvt = o
      ),
      class = c("R7", "R7Private")
    )
  }
  
  if (!is.function(o)) {
    return(o)
  }

  if (is_private) {
    environment(o) <- unclass(self)$env
  } else {
    environment(o) <- rlang::env(.r7_env, self = self, private = self$private)
  }

  if (call && isTRUE(attr(o, "active"))) {
    o()
  } else {
    o
  }
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
