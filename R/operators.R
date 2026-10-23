# An R scalar is a plain length-1 numeric or logical: no dimensions and no
# class. A 1x1 matrix keeps its dimensions, and a classed value such as
# `bit64::integer64` keeps going through `torch_tensor()`.
is_r_scalar <- function(x) {
  (is.numeric(x) || is.logical(x)) && length(x) == 1L && is.null(dim(x)) &&
    !is.object(x)
}

# Convert a non-tensor operand. When libtorch has a Scalar overload for the
# operation the operators below pass the R scalar straight to it, which is
# exactly what `t - 1` already does. This helper covers `%%` and `%/%`: the
# scalar becomes a 0-dim tensor whose dtype is the one libtorch would use for
# `op(ref, x)` with `x` passed as a Scalar, so it takes part in type
# promotion the way a Python scalar does. `5 %% t` keeps `t`'s dtype
# (bfloat16, float64, int64 with `5L`) instead of promoting to float32.
operand_to_tensor <- function(x, ref) {
  if (is_r_scalar(x)) {
    torch_scalar_tensor(x, dtype = torch_result_type(ref, x), device = ref$device)
  } else {
    torch_tensor(x, device = ref$device)
  }
}

# `&` and `|` only need the scalar's truth value, so it becomes a bool tensor.
# Converting to the other operand's dtype could wrap `256L` to zero in uint8
# or underflow `1e-8` to zero in float16.
logical_operand <- function(x, ref) {
  if (is_r_scalar(x)) {
    torch_scalar_tensor(as.logical(x), device = ref$device)
  } else {
    torch_tensor(x, device = ref$device)
  }
}

# True division always gives a floating result, so an integer scalar must not
# be narrowed to the tensor's integer dtype first: `300L / uint8` is 300, not
# 44. The quotient is computed at the output dtype, except that float16 and
# bfloat16 inputs are computed at float32 and cast back afterwards, as
# libtorch's own kernels do per element. Dividing directly at that precision
# keeps a large scalar (`65536 / half`) and a tiny denominator
# (`1e-5 / half(1e-5)`) from overflowing when the quotient itself is
# representable; `reciprocal(t) * x` would overflow `1 / t` in the second case.
# A double scalar outside float32's normal range (`1e40`, `1e-50`) would
# itself overflow or flush to zero at float32, so the quotient is computed at
# float64 then. MPS has no float64, so there it stays at float32.
divide_scalar_by <- function(x, ref) {
  out <- torch_result_type(ref, x)
  if (!out$is_floating_point && !ref$is_complex()) {
    out <- torch_get_default_dtype()
  }
  compute <- out
  if (out == torch_float16() || out == torch_bfloat16()) {
    compute <- torch_float32()
  }
  if (outside_float32_range(x) && ref$device$type != "mps") {
    if (compute == torch_float32()) {
      compute <- torch_float64()
    } else if (compute == torch_cfloat()) {
      compute <- torch_cdouble()
    }
  }
  numerator <- torch_scalar_tensor(x, dtype = compute, device = ref$device)
  torch_div(numerator, ref$to(dtype = compute))$to(dtype = out)
}

# TRUE when a finite nonzero scalar would overflow or become subnormal or zero
# as a float32. The limits are FLT_MAX and FLT_MIN.
outside_float32_range <- function(x) {
  a <- abs(as.double(x))
  is.finite(a) && a != 0 && (a > 3.4028234663852886e38 || a < 1.1754943508222875e-38)
}

#' @export
`+.torch_tensor` <- function(e1, e2) {
  if (missing(e2)) {
    e2 <- torch_zeros_like(e1)
  }

  if (is_r_scalar(e1)) {
    return(torch_add(e2, e1))
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

  if (is_r_scalar(e1)) {
    return(torch_rsub(e2, e1))
  }

  if (!is_torch_tensor(e1)) {
    e1 <- torch_tensor(e1, device = e2$device)
  }

  torch_sub(e1, e2)
}

#' @export
`*.torch_tensor` <- function(e1, e2) {
  if (is_r_scalar(e1)) {
    return(torch_mul(e2, e1))
  }

  if (!is_torch_tensor(e1)) {
    e1 <- torch_tensor(e1, device = e2$device)
  }

  torch_mul(e1, e2)
}

#' @export
`/.torch_tensor` <- function(e1, e2) {
  if (is_r_scalar(e1)) {
    return(divide_scalar_by(e1, e2))
  }

  if (!is_torch_tensor(e1)) {
    e1 <- torch_tensor(e1, device = e2$device)
  }

  torch_div(e1, e2)
}

#' @export
`^.torch_tensor` <- function(e1, e2) {
  # `torch_pow()` has a Scalar-first overload, so an R scalar passes through.
  if (!is_torch_tensor(e1) && !is_r_scalar(e1)) {
    e1 <- torch_tensor(e1, device = e2$device)
  }

  torch_pow(e1, e2)
}

#' @export
`%%.torch_tensor` <- function(e1, e2) {
  if (!is_torch_tensor(e1)) {
    e1 <- operand_to_tensor(e1, e2)
  }

  torch_fmod(e1, e2)
}

#' @export
`%/%.torch_tensor` <- function(e1, e2) {
  if (!is_torch_tensor(e1)) {
    e1 <- operand_to_tensor(e1, e2)
  }

  torch_div(e1, e2, rounding_mode = "trunc")
}

# Comparisons have Scalar overloads, so an R scalar on either side passes
# straight through; anything else that is not a tensor is converted.
comparison_operands <- function(e1, e2) {
  if (!is_torch_tensor(e1) && !is_r_scalar(e1)) {
    e1 <- torch_tensor(e1, device = e2$device)
  }

  if (!is_torch_tensor(e2) && !is_r_scalar(e2)) {
    e2 <- torch_tensor(e2, device = e1$device)
  }

  list(e1, e2)
}

#' @export
`>=.torch_tensor` <- function(e1, e2) {
  if (is_r_scalar(e1)) {
    return(torch_le(e2, e1))
  }

  args <- comparison_operands(e1, e2)
  torch_ge(args[[1]], args[[2]])
}

#' @export
`>.torch_tensor` <- function(e1, e2) {
  if (is_r_scalar(e1)) {
    return(torch_lt(e2, e1))
  }

  args <- comparison_operands(e1, e2)
  torch_gt(args[[1]], args[[2]])
}

#' @export
`<=.torch_tensor` <- function(e1, e2) {
  if (is_r_scalar(e1)) {
    return(torch_ge(e2, e1))
  }

  args <- comparison_operands(e1, e2)
  torch_le(args[[1]], args[[2]])
}

#' @export
`<.torch_tensor` <- function(e1, e2) {
  if (is_r_scalar(e1)) {
    return(torch_gt(e2, e1))
  }

  args <- comparison_operands(e1, e2)
  torch_lt(args[[1]], args[[2]])
}

#' @export
`==.torch_tensor` <- function(e1, e2) {
  if (is_r_scalar(e1)) {
    return(torch_eq(e2, e1))
  }

  args <- comparison_operands(e1, e2)
  torch_eq(args[[1]], args[[2]])
}

#' @export
`!=.torch_tensor` <- function(e1, e2) {
  if (is_r_scalar(e1)) {
    return(torch_ne(e2, e1))
  }

  args <- comparison_operands(e1, e2)
  torch_ne(args[[1]], args[[2]])
}

#' @export
`&.torch_tensor` <- function(e1, e2) {
  if (!is_torch_tensor(e1)) {
    e1 <- logical_operand(e1, e2)
  }

  if (!is_torch_tensor(e2)) {
    e2 <- logical_operand(e2, e1)
  }

  torch_logical_and(e1, e2)
}

#' @export
`|.torch_tensor` <- function(e1, e2) {
  if (!is_torch_tensor(e1)) {
    e1 <- logical_operand(e1, e2)
  }

  if (!is_torch_tensor(e2)) {
    e2 <- logical_operand(e2, e1)
  }

  torch_logical_or(e1, e2)
}

#' @export
`!.torch_tensor` <- function(x) {
  torch_logical_not(x)
}

#' @rawNamespace if (getRversion() >= "4.3.0") S3method("%*%", torch_tensor)
`%*%.torch_tensor` <- function(x, y) {
  if (!is_torch_tensor(y)) {
    y <- torch_tensor(y, device = x$device)
  }
  torch_matmul(x, y)
}

#' @export
dim.torch_tensor <- function(x) {
  cpp_tensor_dim(x$ptr)
}

#' @export
as.numeric.torch_tensor <- function(x, ...) {
  as.numeric(as_array(x))
}

#' @export
as.complex.torch_tensor <- function(x, ...) {
  as.complex(as_array(x))
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
  if (na.rm) value_error("Torch tensors do not have NAs!")
  l <- list(...)
  l_max <- lapply(l, torch_max)
  Reduce(function(x, y) torch_max(x, other = y), l_max)
}

#' @export
min.torch_tensor <- function(..., na.rm = FALSE) {
  if (na.rm) value_error("Torch tensors do not have NAs!")
  l <- list(...)
  l_min <- lapply(l, torch_min)
  Reduce(function(x, y) torch_min(x, other = y), l_min)
}

#' @export
prod.torch_tensor <- function(..., dim, keepdim = FALSE, na.rm = FALSE) {
  if (na.rm) value_error("Torch tensors do not have NAs!")
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
  if (na.rm) value_error("Torch tensors do not have NAs!")
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
  if (na.rm) value_error("Torch tensors do not have NAs!")
  if (!missing(dim)) {
    torch_mean(x, dim, keepdim)
  } else {
    torch_mean(x)
  }
}
