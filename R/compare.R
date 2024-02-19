compare_proxy.torch_tensor <- function(x, path) {
  list(
    object = list(
      x = torch::as_array(x),
      device = as.character(x$device),
      requires_grad = x$requires_grad,
      grad_fn = x$grad_fn
    ),
    path = path
  )
}

# shamelessly copied from: https://github.com/tidyverse/readr/blob/e529cb2775f1b52a0dfa30dabc9f8e0014aa77e6/R/zzz.R
register_s3_method <- function(pkg, generic, class, fun = NULL) {
  if (is.null(fun)) {
    fun <- get(paste0(generic, ".", class), envir = parent.frame())
  } else {
    stopifnot(is.function(fun))
  }

  if (pkg %in% loadedNamespaces()) {
    registerS3method(generic, class, fun, envir = asNamespace(pkg))
  }

  # Always register hook in case package is later unloaded & reloaded
  setHook(
    packageEvent(pkg, "onLoad"),
    function(...) {
      registerS3method(generic, class, fun, envir = asNamespace(pkg))
    }
  )
}

