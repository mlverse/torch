nullptr <- function() {
  x <- cpp_nullptr()
  class(x) <- "nullptr"
  x
}