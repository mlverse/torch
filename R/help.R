# see https://github.com/rstudio/rstudio/pull/780

#' @importFrom utils .DollarNames
#' @export
.DollarNames.torch_tensor <- function(x, pattern = "") {
  candidates <- names(parent.env(Tensor))
  candidates <- sort(candidates[grepl(pattern, candidates)])
  attr(candidates, "helpHandler") <- "torch:::help_handler"
  candidates
}

#' @importFrom utils .DollarNames
#' @export
.DollarNames.nn_module <- function(x, pattern = "") {
  pars <- names(x[[".__enclos_env__"]][["private"]][["parameters_"]])
  bufs <- names(x[[".__enclos_env__"]][["private"]][["buffers_"]])
  mods <- names(x[[".__enclos_env__"]][["private"]][["modules_"]])
  candidates <- unique(c(pars, bufs, mods, names(x)))
  candidates <- sub("(^[0-9]+.*)", replacement = "`\\1`", candidates)
  candidates <- sort(candidates[grepl(pattern, candidates)])
  candidates
}

help_formals_handler.torch_tensor <- function(topic, source, ...) {
  list(
    formals = get_tensor_method_arguments(topic),
    helpHandler = "torch:::help_handler"
  )
}

help_handler <- function(type, topic, source, ...) {
  if (type == "completion") {
    help_handler_completion(topic, source, ...)
  } else if (type == "parameter") {
    help_handler_parameter(topic, source, ...)
  } else {
    NULL
  }
}

get_tensor_method_arguments <- function(topic) {
  tryCatch(
    rlang::fn_fmls_names(parent.env(Tensor)[[topic]])[-c(1, 2)],
    error = function(e) {
      NULL
    }
  )
}

help_handler_completion <- function(topic, source, ...) {
  signature <- get_tensor_method_arguments(topic)
  signature <- paste0(signature, collapse = ", ")
  signature <- paste0(topic, "(", signature, ")")

  list(title = topic, signature = signature)
}

help_handler_parameter <- function(topic, source, ...) {
  list(
    args = get_tensor_method_arguments(topic)
  )
}
