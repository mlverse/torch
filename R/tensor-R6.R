`torch::Tensor` <- R6::R6Class(
  "tensor",
  cloneable = FALSE,

  private = list(
    xp = NULL
  ),

  active = list(
    pointer = function(xp) {
      if (missing(xp)) {
        private$xp
      } else {
        stop("Pointer is read-only!")
      }
    }
  ),

  public = list(

    initialize = function (xp) {
      private$xp <- xp
    },

    print = function (...){
      cat(sprintf("%s", class(self)[[1]]), "\n")
      tensor_print_(self$pointer)
      invisible(self)
    },

    as_vector = function () {

      a <- as_array_tensor_(self$pointer)

      if (length(a$dim) <= 1L) {
        out <- a$vec
      } else if (length(a$dim) == 2L) {
        out <- t(matrix(a$vec, ncol = a$dim[1], nrow = a$dim[2]))
      } else {
        out <- aperm(array(a$vec, dim = rev(a$dim)), seq(length(a$dim), 1))
      }

      out

    }

  )

)


external_ptr <- function(class, xp) {
  class$new(xp)
}

`torch::Tensor`$dispatch <- function(xp){
  external_ptr(`torch::Tensor`, xp)
}
