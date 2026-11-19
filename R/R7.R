prepare_method <- function(m, active = FALSE) {
  if (is.function(m)) {
    formals(m) <- c(alist(self = , private = ), formals(m))  
  }
  if (active) {
    attr(m, "active") <- TRUE
  }
  m
}

.generators <- new.env(parent = emptyenv())
.r7_pvt_class <- c("R7_private", "R7")

#' @importFrom rlang env_bind
#' @importFrom rlang :=
R7Class <- function(classname = NULL, public = list(), private = list(),
                    active = list()) {
  methods <- new.env(parent = emptyenv())
  private_methods <- new.env(parent = emptyenv())

  public <- lapply(public, prepare_method)
  active <- lapply(active, prepare_method, active = TRUE)
  private <- lapply(private, prepare_method)

  env_bind(methods, !!!public)
  env_bind(methods, !!!active)
  env_bind(private_methods, !!!private)
  methods$private <- private_methods

  generator <- new.env(parent = methods)

  generator$new <- function(...) {
    self <- methods$initialize(NULL, NULL, ...)
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
      value_error("can only set to `public`, `private` and `active`")
    }
  }
  
  generator$methods <- methods

  # set the generator/classname env
  .generators[[classname]] <- generator

  generator
}

r7_func_factory <- function(self, private, fn) {
  f <- function(...) {
    fn(self, private,...)
  }
  attr(f, "active") <- attr(fn, "active")
  f
}

extract_method <- function(self, name, call = TRUE) {
  if (inherits(self, "R7_private")) {
    self <- attr(self, "r7_slf")
    method <- .generators[[class(self)[1]]][["methods"]][["private"]][[name]]
  } else {
    if (name == "private") {
      return(structure(list(), r7_slf = self, class = .r7_pvt_class))
    }
    method <- .generators[[class(self)[1]]][["methods"]][[name]]
  }

  if (call && isTRUE(attr(method, "active"))) {
    method(self, self$private)
  } else {
    r7_func_factory(self, self$private, method)
  }
}

#' @export
`$.R7` <- extract_method

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

#' @export
length.R7 <- function(x) {
  tryCatch(
    x$length(), 
    error = function(err) {
      1 # when no custom length method is implemented, we return 1.
    }
  )
}