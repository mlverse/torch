FunctionSchema <- R7Class(
  classname = "function_schema",
  public = list(
    print = function() {
      cat(paste0("function_schema (name = ", self$name, ")\n"))
      cat(gettext("arguments: "))
      if (!is.null(names(self$arguments))) {
        cat(gettextf("%s \n", paste(names(self$arguments), self$arguments, sep = " - ", collapse = ", ")))
      } else {
        cat(gettextf("%s \n", paste(self$arguments, collapse = ", ")))
      }
      cat(gettext("returns: "))
      cat(gettextf("%s \n", paste(self$returns, collapse = ", ")))
    }
  ),
  active = list(
    name = function() {
      name <- function_schema_name(self);
      name
    },
    arguments = function() {
      arguments <- function_schema_arguments(self)
      num_args <- length(arguments)
      if (num_args > 0) {
        arglist <- vector(mode = "list", length = num_args)
        argnames <- vector(mode = "character", length = num_args)
        for (i in 1:num_args) {
          name <- function_schema_argument_name(arguments[[i]])
          type <- function_schema_argument_type(arguments[[i]])
          arglist[[i]] <- type
          argnames[[i]] <- name
        }
        names(arglist) <- argnames
      } else {
        arglist <- vector(mode = "list", length = 1)
        arglist[[1]] <- "<none>"
      }
      arglist
    },
    returns = function() {
      returns <- function_schema_returns(self)
      num_returns <- length(returns)
      if (num_returns > 0) {
        retlist <- vector(mode = "list", length = num_returns)
        for (i in 1:num_returns) {
          type <- function_schema_argument_type(returns[[i]])
          retlist[[i]] <- type
        }
      } else {
        retlist <- vector(mode = "list", length = 1)
        retlist[[1]] <- "<none>"
      }
      retlist
    }
  )
)

