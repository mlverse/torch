Dimname <- R7Class(
  classname = "torch_dimname",
  public = list(
    ptr = NULL,
    initialize = function(name, ptr = NULL) {
      if (!is.null(ptr)) {
        return(ptr)
      }

      cpp_torch_dimname(name)
    },
    print = function() {
      print(cpp_dimname_to_string(self$ptr))
    }
  ),
  active = list(
    ptr = function() {
      self
    }
  )
)

torch_dimname <- function(name) {
  Dimname$new(name)
}

is_torch_dimname <- function(x) {
  inherits(x, "torch_dimname")
}

DimnameList <- R7Class(
  classname = "torch_dimname_list",
  public = list(
    ptr = NULL,
    initialize = function(names, ptr = NULL) {
      if (!is.null(ptr)) {
        return(ptr)
      }

      ptrs <- lapply(lapply(names, torch_dimname), function(self) self$ptr)
      cpp_torch_dimname_list(ptrs)
    },
    print = function() {
      print(self$to_r())
    },
    to_r = function() {
      cpp_dimname_list_to_string(self$ptr)
    }
  ),
  active = list(
    ptr = function() {
      self
    }
  )
)

torch_dimname_list <- function(names) {
  DimnameList$new(names)
}

#' @export
as.character.torch_dimname_list <- function(x, ...) {
  cpp_dimname_list_to_string(x$ptr)
}

#' @export
as.character.torch_dimname <- function(x, ...) {
  cpp_dimname_to_string(x$ptr)
}

is_torch_dimname_list <- function(x) {
  inherits(x, "torch_dimname_list")
}
