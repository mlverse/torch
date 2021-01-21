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

