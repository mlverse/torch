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

add_class_definition <- function(r6_class_generator) {
  .new <- r6_class_generator$new
  .class_def <- r6_class_generator
  .wrapped_new <- function(...) {
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
                         attr_name, constructor_class = NULL) {
  args <- list(...)

  if (!is.null(attr(inherit, attr_name))) {
    inherit <- attr(inherit, attr_name)
  }

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
  if (!is.null(constructor_class)) {
    class(f) <- constructor_class
  }

  f
}

# https://stackoverflow.com/a/54971834/3297472
transpose_list <- function(x) {
  do.call(Map, c(f = c, x))
}

#' @importFrom utils head
#' @importFrom utils tail
head2 <- function(x, n) {
  if (n > 0) {
    utils::head(x, n = n)
  } else {
    utils::head(x, n = length(x) + n)
  }
}

seq2 <- function(start, end, by = 1L) {
  if ((end - start) < by && (end - start != 0)) {
    return(integer())
  } else {
    seq(start, end, by = by)
  }
}

torch_option <- function(option, default = NULL) {
  getOption(paste0("torch.", option), default)
}

math_to_rd_impl <- function(tex, ascii = tex, displayMode = TRUE, ..., include_css = TRUE) {
  html <- katex::katex_html(tex,
    include_css = include_css, displayMode = displayMode, ...,
    preview = FALSE
  )

  html_out <- paste("\\if{html}{\\out{", html, "}}", sep = "\n")
  # We won't show the equations in latex mode because of limitted support.
  latex_out <- paste("\\if{latex,text}{\\out{", ascii, "}}", sep = "\n")
  rd <- paste(html_out, latex_out, sep = "\n")
  if (identical(.Platform$OS.type, "windows") && getRversion() < "4.1.1") {
    # https://bugs.r-project.org/bugzilla/show_bug.cgi?id=18152
    rd <- enc2native(rd)
  }
  structure(rd, class = "Rdtext")
}

math_to_rd <- function(tex, ascii = tex, displayMode = TRUE, ..., include_css = TRUE) {
  if (tex == ascii) {
    ascii <- "Equation not displayed. Please find it in 'https://torch.mlverse.org/docs'"
  }

  if (rlang::is_installed("katex")) {
    math_to_rd_impl(tex, ascii, displayMode = displayMode, ..., include_css = include_css)
  } else {
    rd <- "\\out{Equation not displayed. Install 'katex' then re-install 'torch'.}"
    structure(rd, class = "Rdtext")
  }
}
