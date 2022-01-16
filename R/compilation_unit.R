CompilationUnit <- R7Class(
  classname = "torch_compilation_unit",
  public = list(
    ptr = NULL,
    initialize = function() {
      cpp_jit_compilation_unit()
    },
    print = function() {
      cat("<torch_compilation_unit>")
    }
  )
)

#' @export
`$.torch_compilation_unit` <- function(x, name) {
  f <- cpp_jit_compile_get_function(x, name)
  if (is.null(f)) {
    return(NextMethod("$", x))
  }
  new_script_function(f)
}

#' @export
.DollarNames.torch_compilation_unit <- function(x, pattern = "") {
  candidates <- cpp_jit_compile_list_methods(x)
  candidates <- sort(candidates[grepl(pattern, candidates)])
  # attr(candidates, "helpHandler") <- "torch:::help_handler"
  candidates
}
