nullptr <- function() {
  x <- cpp_nullptr()
  class(x) <- "nullptr"
  x
}

# https://stackoverflow.com/a/27350487/3297472
is_null_external_pointer <- function(pointer) {
  a <- attributes(pointer)
  attributes(pointer) <- NULL
  out <- identical(pointer, methods::new("externalptr"))
  attributes(pointer) <- a
  out
}

add_class_definition <- function(r6_class_generator){
  .new <- r6_class_generator$new
  .class_def <- r6_class_generator
  .wrapped_new <- function(...){
    .object <- .new(...)
    .object$.__enclos_env__$self$class_def <- .class_def
    .object
  }
  r6_class_generator$unlock()
  r6_class_generator$new <- .wrapped_new
  r6_class_generator$lock()
  r6_class_generator
}

create_class <- function(name, inherit, ..., private, active, parent_env,
                         attr_name) {
  
  args <- list(...)
  
  if (!is.null(attr(inherit, attr_name)))
    inherit <- attr(inherit, attr_name)
  
  e <- new.env(parent = parent_env)
  e$inherit <- inherit
  
  d <- R6::R6Class(
    classname = name,
    lock_objects = FALSE,
    inherit = inherit,
    public = args,
    private = private,
    active = active,
    parent_env = e
  )
  
  init <- get_init(d)
  # same signature as the init method, but calls with dataset$new.
  f <- rlang::new_function(
    args = rlang::fn_fmls(init),
    body = rlang::expr({
      d$new(!!!rlang::fn_fmls_syms(init))
    })
  )
  
  attr(f, attr_name) <- d
  f
}

# https://stackoverflow.com/a/54971834/3297472
transpose_list <- function(x) {
  do.call(Map, c(f = c, x))
}
  
head2 <- function(x, n) {
  if (n > 0)
    utils::head(x, n = n)
  else
    utils::head(x, n = length(x) + n)
}

