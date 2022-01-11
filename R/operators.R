#' @export
`+.torch_tensor` <- function(e1, e2) {
  if (missing(e2)) {
    e2 <- torch_zeros_like(e1)
  }

  if (!is_torch_tensor(e1)) {
    e1 <- torch_tensor(e1, device = e2$device)
  }

  torch_add(e1, e2)
}

#' @export
`-.torch_tensor` <- function(e1, e2) {
  if (missing(e2)) {
    e2 <- e1
    e1 <- torch_zeros_like(e1)
  }

  if (!is_torch_tensor(e1)) {
    e1 <- torch_tensor(e1, device = e2$device)
  }

  torch_sub(e1, e2)
}

#' @export
`*.torch_tensor` <- function(e1, e2) {
  if (!is_torch_tensor(e1)) {
    e1 <- torch_tensor(e1, device = e2$device)
  }

  torch_mul(e1, e2)
}

#' @export
`/.torch_tensor` <- function(e1, e2) {
  if (!is_torch_tensor(e1)) {
    e1 <- torch_tensor(e1, device = e2$device)
  }

  torch_div(e1, e2)
}

#' @export
`^.torch_tensor` <- function(e1, e2) {
  if (!is_torch_tensor(e1)) {
    e1 <- torch_tensor(e1, device = e2$device)
  }

  torch_pow(e1, e2)
}

#' @export
`%%.torch_tensor` <- function(e1, e2) {
  if (!is_torch_tensor(e1)) {
    e1 <- torch_tensor(e1, device = e2$device)
  }

  torch_fmod(e1, e2)
}

#' @export
`%/%.torch_tensor` <- function(e1, e2) {
  if (!is_torch_tensor(e1)) {
    e1 <- torch_tensor(e1, device = e2$device)
  }

  torch_div(e1, e2, rounding_mode = "trunc")
}

#' @export
`>=.torch_tensor` <- function(e1, e2) {
  if (!is_torch_tensor(e1)) {
    e1 <- torch_tensor(e1, device = e2$device)
  }

  if (!is_torch_tensor(e2)) {
    e2 <- torch_tensor(e2, device = e1$device)
  }

  torch_ge(e1, e2)
}

#' @export
`>.torch_tensor` <- function(e1, e2) {
  if (!is_torch_tensor(e1)) {
    e1 <- torch_tensor(e1, device = e2$device)
  }

  if (!is_torch_tensor(e2)) {
    e2 <- torch_tensor(e2, device = e1$device)
  }

  torch_gt(e1, e2)
}

#' @export
`<=.torch_tensor` <- function(e1, e2) {
  if (!is_torch_tensor(e1)) {
    e1 <- torch_tensor(e1, device = e2$device)
  }

  if (!is_torch_tensor(e2)) {
    e2 <- torch_tensor(e2, device = e1$device)
  }

  torch_le(e1, e2)
}

#' @export
`<.torch_tensor` <- function(e1, e2) {
  if (!is_torch_tensor(e1)) {
    e1 <- torch_tensor(e1, device = e2$device)
  }

  if (!is_torch_tensor(e2)) {
    e2 <- torch_tensor(e2, device = e1$device)
  }

  torch_lt(e1, e2)
}

#' @export
`==.torch_tensor` <- function(e1, e2) {
  if (!is_torch_tensor(e1)) {
    e1 <- torch_tensor(e1, device = e2$device)
  }

  if (!is_torch_tensor(e2)) {
    e2 <- torch_tensor(e2, device = e1$device)
  }

  torch_eq(e1, e2)
}

#' @export
`!=.torch_tensor` <- function(e1, e2) {
  if (!is_torch_tensor(e1)) {
    e1 <- torch_tensor(e1, device = e2$device)
  }

  torch_ne(e1, e2)
}

#' @export
`&.torch_tensor` <- function(e1, e2) {
  if (!is_torch_tensor(e1)) {
    e1 <- torch_tensor(e1, device = e2$device)
  }

  if (!is_torch_tensor(e2)) {
    e2 <- torch_tensor(e2, device = e1$device)
  }

  torch_logical_and(e1, e2)
}

#' @export
`|.torch_tensor` <- function(e1, e2) {
  if (!is_torch_tensor(e1)) {
    e1 <- torch_tensor(e1, device = e2$device)
  }

  if (!is_torch_tensor(e2)) {
    e2 <- torch_tensor(e2, device = e1$device)
  }

  torch_logical_or(e1, e2)
}

#' @export
`!.torch_tensor` <- function(x) {
  torch_logical_not(x)
}

#' @export
dim.torch_tensor <- function(x) {
  cpp_tensor_dim(x$ptr)
}

#' @export
length.torch_tensor <- function(x) {
  prod(dim(x))
}

#' @export
as.numeric.torch_tensor <- function(x, ...) {
  as.numeric(as_array(x))
}

#' @export
as.integer.torch_tensor <- function(x, ...) {
  as.integer(as_array(x))
}

#' @export
as.logical.torch_tensor <- function(x, ...) {
  as.logical(as_array(x))
}

#' @export
as.double.torch_tensor <- function(x, ...) {
  as.double(as_array(x))
}

#' @export
abs.torch_tensor <- function(x) {
  torch_abs(x)
}

#' @export
sign.torch_tensor <- function(x) {
  torch_sign(x)
}

#' @export
sqrt.torch_tensor <- function(x) {
  torch_sqrt(x)
}

#' @export
ceiling.torch_tensor <- function(x) {
  torch_ceil(x)
}

#' @export
floor.torch_tensor <- function(x) {
  torch_floor(x)
}

#' @export
trunc.torch_tensor <- function(x, ...) {
  torch_trunc(x)
}

#' @export
cumsum.torch_tensor <- function(x) {
  torch_cumsum(x, dim = 1)
}

#' @export
log.torch_tensor <- function(x, base) {
  if (!missing(base)) {
    torch_log(x) / torch_log(base)
  } else {
    torch_log(x)
  }
}

#' @method log10 torch_tensor
#' @export
log10.torch_tensor <- function(x) {
  torch_log10(x)
}

#' @method log2 torch_tensor
#' @export
log2.torch_tensor <- function(x) {
  torch_log2(x)
}

#' @export
log1p.torch_tensor <- function(x) {
  torch_log1p(x)
}

#' @export
acos.torch_tensor <- function(x) {
  torch_acos(x)
}

#' @export
asin.torch_tensor <- function(x) {
  torch_asin(x)
}

#' @export
atan.torch_tensor <- function(x) {
  torch_atan(x)
}

#' @export
exp.torch_tensor <- function(x) {
  torch_exp(x)
}

#' @export
expm1.torch_tensor <- function(x) {
  torch_expm1(x)
}

#' @export
cos.torch_tensor <- function(x) {
  torch_cos(x)
}

#' @export
cosh.torch_tensor <- function(x) {
  torch_cosh(x)
}

#' @export
sin.torch_tensor <- function(x) {
  torch_sin(x)
}

#' @export
sinh.torch_tensor <- function(x) {
  torch_sinh(x)
}

#' @export
tan.torch_tensor <- function(x) {
  torch_tan(x)
}

#' @export
tanh.torch_tensor <- function(x) {
  torch_tanh(x)
}

#' @export
max.torch_tensor <- function(..., na.rm = FALSE) {
  if (na.rm) stop("Torch tensors do not have NAs!")
  l <- list(...)
  l_max <- lapply(l, torch_max)
  Reduce(function(x, y) torch_max(x, other = y), l_max)
}

#' @export
min.torch_tensor <- function(..., na.rm = FALSE) {
  if (na.rm) stop("Torch tensors do not have NAs!")
  l <- list(...)
  l_min <- lapply(l, torch_min)
  Reduce(function(x, y) torch_min(x, other = y), l_min)
}

#' @export
prod.torch_tensor <- function(..., dim, keepdim = FALSE, na.rm = FALSE) {
  if (na.rm) stop("Torch tensors do not have NAs!")
  l <- list(...)
  if (length(l) == 1) {
    if (!missing(dim)) {
      return(torch_prod(l[[1]], dim = dim, keepdim = keepdim))
    } else {
      return(torch_prod(l[[1]]))
    }
  } else {
    stopifnot(missing(dim))
    return(Reduce(`*`, lapply(l, torch_prod)))
  }
}

#' @export
sum.torch_tensor <- function(..., dim, keepdim = FALSE, na.rm = FALSE) {
  if (na.rm) stop("Torch tensors do not have NAs!")
  l <- list(...)
  if (length(l) == 1) {
    if (!missing(dim)) {
      return(torch_sum(l[[1]], dim = dim, keepdim = keepdim))
    } else {
      return(torch_sum(l[[1]]))
    }
  } else {
    stopifnot(missing(dim))
    return(Reduce(`+`, lapply(l, torch_sum)))
  }
}

#' @export
mean.torch_tensor <- function(x, dim, keepdim = FALSE, na.rm = FALSE, ...) {
  if (na.rm) stop("Torch tensors do not have NAs!")
  if (!missing(dim)) {
    torch_mean(x, dim, keepdim)
  } else {
    torch_mean(x)
  }
}
