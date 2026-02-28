torch_dtype <- R7Class(
  classname = "torch_dtype",
  public = list(
    ptr = NULL,
    initialize = function(ptr = NULL) {
      ptr
    },
    print = function() {
      cat("torch_", self$.type(), "\n", sep = "")
    },
    .type = function() {
      cpp_dtype_to_string(self$ptr)
    }
  ),
  active = list(
    is_floating_point = function() {
      if (cpp_dtype_to_string(self$ptr) %in% c("Float", "Double", "Half", "BFloat16", "Float8_e4m3fn", "Float8_e5m2")) {
        TRUE
      } else {
        FALSE
      }
    },
    ptr = function() {
      self
    }
  )
)

#' @export
as.character.torch_dtype <- function(x, ...) {
  cpp_dtype_to_string(x$ptr)
}

dtype_from_string <- function(str) {
  switch(tolower(str),
    "float" = torch_float(),
    "float32" = torch_float32(),
    "float64" = torch_float64(),
    "double" = torch_double(),
    "float16" = torch_float16(),
    "half" = torch_half(),
    "uint8" = torch_uint8(),
    "int8" = torch_int8(),
    "int16" = torch_int16(),
    "short" = torch_short(),
    "int32" = torch_int32(),
    "int" = torch_int(),
    "int64" = torch_int64(),
    "long" = torch_long(),
    "bool" = torch_bool(),
    "quint8" = torch_quint8(),
    "qint8" = torch_qint8(),
    "qint32" = torch_qint32(),
    "chalf" = torch_chalf(),
    "cfloat" = torch_cfloat(),
    "cdouble" = torch_cdouble(),
    "float8_e4m3fn" = torch_float8_e4m3fn(),
    "float8_e5m2" = torch_float8_e5m2()
  )
}


#' Torch data types
#'
#' Returns the correspondent data type.
#'
#' @name torch_dtype
#' @rdname torch_dtype
#' @concept tensor-attributes
#'
NULL

#' @rdname torch_dtype
#' @export
torch_float32 <- function() torch_dtype$new(cpp_torch_float32())
#' @rdname torch_dtype
#' @export
torch_float <- function() torch_dtype$new(cpp_torch_float32())
#' @rdname torch_dtype
#' @export
torch_float64 <- function() torch_dtype$new(cpp_torch_float64())
#' @rdname torch_dtype
#' @export
torch_double <- function() torch_dtype$new(cpp_torch_float64())


#' @rdname torch_dtype
#' @export
torch_cfloat32 <- function() torch_dtype$new(cpp_torch_chalf())
#' @rdname torch_dtype
#' @export
torch_chalf <- function() torch_dtype$new(cpp_torch_chalf())

#' @rdname torch_dtype
#' @export
torch_cfloat <- function() torch_dtype$new(cpp_torch_cfloat())
#' @rdname torch_dtype
#' @export
torch_cfloat64 <- function() torch_dtype$new(cpp_torch_cfloat())

#' @rdname torch_dtype
#' @export
torch_cdouble <- function() torch_dtype$new(cpp_torch_cdouble())
#' @rdname torch_dtype
#' @export
torch_cfloat128 <- function() torch_dtype$new(cpp_torch_cdouble())

#' @rdname torch_dtype
#' @export
torch_float16 <- function() torch_dtype$new(cpp_torch_float16())
#' @rdname torch_dtype
#' @export
torch_half <- function() torch_dtype$new(cpp_torch_float16())
#' @rdname torch_dtype
#' @export
torch_bfloat16 <- function() torch_dtype$new(cpp_torch_bfloat16())

#' @rdname torch_dtype
#' @export
torch_float8_e4m3fn <- function() torch_dtype$new(cpp_torch_float8_e4m3fn())
#' @rdname torch_dtype
#' @export
torch_float8_e5m2 <- function() torch_dtype$new(cpp_torch_float8_e5m2())

#' @rdname torch_dtype
#' @export
torch_uint8 <- function() torch_dtype$new(cpp_torch_uint8())

#' @rdname torch_dtype
#' @export
torch_int8 <- function() torch_dtype$new(cpp_torch_int8())

#' @rdname torch_dtype
#' @export
torch_int16 <- function() torch_dtype$new(cpp_torch_int16())
#' @rdname torch_dtype
#' @export
torch_short <- function() torch_dtype$new(cpp_torch_int16())

#' @rdname torch_dtype
#' @export
torch_int32 <- function() torch_dtype$new(cpp_torch_int32())
#' @rdname torch_dtype
#' @export
torch_int <- function() torch_dtype$new(cpp_torch_int32())

#' @rdname torch_dtype
#' @export
torch_int64 <- function() torch_dtype$new(cpp_torch_int64())
#' @rdname torch_dtype
#' @export
torch_long <- function() torch_dtype$new(cpp_torch_int64())

#' @rdname torch_dtype
#' @export
torch_bool <- function() torch_dtype$new(cpp_torch_bool())

#' @rdname torch_dtype
#' @export
torch_quint8 <- function() torch_dtype$new(cpp_torch_quint8())

#' @rdname torch_dtype
#' @export
torch_qint8 <- function() torch_dtype$new(cpp_torch_qint8())

#' @rdname torch_dtype
#' @export
torch_qint32 <- function() torch_dtype$new(cpp_torch_qint32())


#' @export
`==.torch_dtype` <- function(e1, e2) {
  if (!is_torch_dtype(e1) || !is_torch_dtype(e2)) {
    runtime_error("One of the objects is not a dtype. Comparison is not possible.")
  }
  cpp_dtype_to_string(e1$ptr) == cpp_dtype_to_string(e2$ptr)
}

#' @export
`!=.torch_dtype` <- function(e1, e2) {
  !(e1 == e2)
}

#' Check if object is a torch data type
#'
#' @param x object to check.
#' @concept tensor-attributes
#'
#' @export
is_torch_dtype <- function(x) {
  inherits(x, "torch_dtype")
}

is_integer_dtype <- function(x) {
  is_torch_dtype(x) && (cpp_dtype_to_string(x) %in% c("Int", "Short", "Long", "Char", "Byte"))
}

#' Gets and sets the default floating point dtype.
#'
#' @param d The default floating point dtype to set. Initially set to
#'   `torch_float()`.
#'
#' @rdname default_dtype
#' @concept tensor-attributes
#'
#' @export
torch_set_default_dtype <- function(d) {
  cpp_set_default_dtype(d$ptr)
}

#' @rdname default_dtype
#' @concept tensor-attributes
#' @export
torch_get_default_dtype <- function() {
  torch_dtype$new(cpp_get_default_dtype())
}
