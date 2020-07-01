nullptr <- function() {
  x <- cpp_nullptr()
  class(x) <- "nullptr"
  x
}

# https://stackoverflow.com/a/27350487/3297472
is_null_external_pointer <- function(pointer) {
  a <- attributes(pointer)
  attributes(pointer) <- NULL
  out <- identical(pointer, new("externalptr"))
  attributes(pointer) <- a
  out
}
