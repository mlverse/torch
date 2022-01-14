nn_init_no_grad_uniform <- function(tensor, a, b) {
  with_no_grad({
    out <- tensor$uniform_(a, b)
  })
  out
}

nn_init_no_grad_normal <- function(tensor, mean, std) {
  with_no_grad({
    out <- tensor$normal_(mean, std)
  })
  out
}

nn_init_no_grad_trunc_normal <- function(tensor, mean, std, a, b) {
  if (mean < (a - 2 * std) && mean > (b + 2 * std)) {
    warn(
      "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. ",
      "The distribution of values may be incorrect."
    )
  }

  with_no_grad({
    l <- stats::pnorm((a - mean) / std)
    u <- stats::pnorm((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor$uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor$erfinv_()

    # Transform to proper mean, std
    tensor$mul_(std * sqrt(2))
    tensor$add_(mean)

    # Clamp to ensure it's in the proper range
    tensor$clamp_(min = a, max = b)
  })

  tensor
}

nn_init_no_grad_fill <- function(tensor, val) {
  with_no_grad({
    tensor$fill_(val)
  })
  tensor
}

nn_init_no_grad_zero <- function(tensor) {
  with_no_grad({
    tensor$zero_()
  })
  tensor
}


#' Calculate gain
#'
#' Return the recommended gain value for the given nonlinearity function.
#'
#' @param nonlinearity the non-linear function
#' @param param optional parameter for the non-linear function
#'
#' @export
nn_init_calculate_gain <- function(nonlinearity, param = NULL) {
  linear_fns <- c(
    "linear", "conv1d", "conv2d", "conv3d", "conv_transpose1d",
    "conv_transpose2d", "conv_transpose3d"
  )

  if (nonlinearity %in% linear_fns || nonlinearity == "sigmoid") {
    return(1)
  } else if (nonlinearity == "tanh") {
    return(5 / 3)
  } else if (nonlinearity == "relu") {
    return(sqrt(2))
  } else if (nonlinearity == "leaky_relu") {
    if (is.null(param)) {
      negative_slope <- 0.01
    } else {
      negative_slope <- param
    }
    return(sqrt(2 / (1 + negative_slope^2)))
  } else {
    not_implemented_error("Unsupported nonlinearity: {nonlinearity}")
  }
}

#' Uniform initialization
#'
#' Fills the input Tensor with values drawn from the uniform distribution
#'
#' @param tensor an n-dimensional Tensor
#' @param a the lower bound of the uniform distribution
#' @param b the upper bound of the uniform distribution
#'
#' @examples
#' w <- torch_empty(3, 5)
#' nn_init_uniform_(w)
#' @export
nn_init_uniform_ <- function(tensor, a = 0, b = 1) {
  nn_init_no_grad_uniform(tensor, a, b)
}

#' Normal initialization
#'
#' Fills the input Tensor with values drawn from the normal distribution
#'
#' @inheritParams nn_init_uniform_
#' @param mean the mean of the normal distribution
#' @param std the standard deviation of the normal distribution
#'
#' @examples
#' w <- torch_empty(3, 5)
#' nn_init_normal_(w)
#' @export
nn_init_normal_ <- function(tensor, mean = 0, std = 1) {
  nn_init_no_grad_normal(tensor, mean, std)
}

#' Truncated normal initialization
#'
#' Fills the input Tensor with values drawn from a truncated
#' normal distribution.
#'
#' @inheritParams nn_init_normal_
#' @param a the minimum cutoff value
#' @param b the maximum cutoff value
#'
#' @examples
#' w <- torch_empty(3, 5)
#' nn_init_trunc_normal_(w)
#' @export
nn_init_trunc_normal_ <- function(tensor, mean = 0, std = 1, a = -2, b = 2) {
  nn_init_no_grad_trunc_normal(tensor, mean, std, a, b)
}

#' Constant initialization
#'
#' Fills the input Tensor with the value `val`.
#'
#' @param tensor an n-dimensional `Tensor`
#' @param val the value to fill the tensor with
#'
#' @examples
#' w <- torch_empty(3, 5)
#' nn_init_constant_(w, 0.3)
#' @export
nn_init_constant_ <- function(tensor, val) {
  nn_init_no_grad_fill(tensor, val)
}

#' Ones initialization
#'
#' Fills the input Tensor with the scalar value `1`
#'
#' @param tensor an n-dimensional `Tensor`
#'
#' @examples
#' w <- torch_empty(3, 5)
#' nn_init_ones_(w)
#' @export
nn_init_ones_ <- function(tensor) {
  nn_init_no_grad_fill(tensor, 1)
}

#' Zeros initialization
#'
#' Fills the input Tensor with the scalar value `0`
#'
#' @param tensor an n-dimensional tensor
#'
#' @examples
#' w <- torch_empty(3, 5)
#' nn_init_zeros_(w)
#' @export
nn_init_zeros_ <- function(tensor) {
  nn_init_no_grad_zero(tensor)
}


#' Eye initialization
#'
#' Fills the 2-dimensional input `Tensor` with the identity matrix.
#' Preserves the identity of the inputs in `Linear` layers, where as
#' many inputs are preserved as possible.
#'
#' @param tensor a 2-dimensional torch tensor.
#'
#' @examples
#' w <- torch_empty(3, 5)
#' nn_init_eye_(w)
#' @export
nn_init_eye_ <- function(tensor) {
  with_no_grad({
    size <- tensor$size()
    torch_eye_out(tensor, size[1], size[2])
  })
}

#' Dirac initialization
#'
#' Fills the {3, 4, 5}-dimensional input `Tensor` with the Dirac
#' delta function. Preserves the identity of the inputs in `Convolutional`
#' layers, where as many input channels are preserved as possible. In case
#' of groups>1, each group of channels preserves identity.
#'
#' @param tensor a {3, 4, 5}-dimensional `torch.Tensor`
#' @param groups (optional) number of groups in the conv layer (default: 1)
#'
#' @examples
#' \dontrun{
#' w <- torch_empty(3, 16, 5, 5)
#' nn_init_dirac_(w)
#' }
#'
#' @export
nn_init_dirac_ <- function(tensor, groups = 1) {
  sizes <- tensor$size()
  dimensions <- length(sizes)

  out_chans_per_grp <- floor(sizes[1] / groups)
  min_dim <- min(out_chans_per_grp, sizes[2])

  stop("not implemented")
}

nn_init_calculate_fan_in_and_fan_out <- function(tensor) {
  dimensions <- tensor$dim()
  num_input_fmaps <- tensor$size(2)
  num_output_fmaps <- tensor$size(1)
  receptive_field_size <- 1

  if (dimensions > 2) {
    receptive_field_size <- tensor[1, 1, ..]$numel()
  }

  fan_in <- num_input_fmaps * receptive_field_size
  fan_out <- num_output_fmaps * receptive_field_size

  list(fan_in, fan_out)
}


#' Xavier uniform initialization
#'
#' Fills the input `Tensor` with values according to the method
#' described in `Understanding the difficulty of training deep feedforward
#' neural networks` - Glorot, X. & Bengio, Y. (2010), using a uniform
#' distribution.
#'
#' @param tensor an n-dimensional `Tensor`
#' @param gain an optional scaling factor
#'
#' @examples
#' w <- torch_empty(3, 5)
#' nn_init_xavier_uniform_(w)
#' @export
nn_init_xavier_uniform_ <- function(tensor, gain = 1) {
  fans <- nn_init_calculate_fan_in_and_fan_out(tensor)
  fan_in <- fans[[1]]
  fan_out <- fans[[2]]
  std <- gain * sqrt(2.0 / (fan_in + fan_out))
  a <- sqrt(3.0) * std # Calculate uniform bounds from standard deviation
  nn_init_no_grad_uniform(tensor, -a, a)
}

#' Xavier normal initialization
#'
#' Fills the input `Tensor` with values according to the method
#' described in `Understanding the difficulty of training deep feedforward
#' neural networks` - Glorot, X. & Bengio, Y. (2010), using a normal
#' distribution.
#'
#' @param tensor an n-dimensional `Tensor`
#' @param gain an optional scaling factor
#'
#' @examples
#' w <- torch_empty(3, 5)
#' nn_init_xavier_normal_(w)
#' @export
nn_init_xavier_normal_ <- function(tensor, gain = 1) {
  fans <- nn_init_calculate_fan_in_and_fan_out(tensor)
  fan_in <- fans[[1]]
  fan_out <- fans[[2]]
  std <- gain * sqrt(2.0 / (fan_in + fan_out))
  nn_init_no_grad_normal(tensor, 0, std)
}

nn_init_calculate_correct_fan <- function(tensor, mode) {
  mode <- tolower(mode)

  fans <- nn_init_calculate_fan_in_and_fan_out(tensor)
  fan_in <- fans[[1]]
  fan_out <- fans[[2]]

  if (mode == "fan_in") {
    fan_in
  } else {
    fan_out
  }
}

#' Kaiming uniform initialization
#'
#' Fills the input `Tensor` with values according to the method
#' described in `Delving deep into rectifiers: Surpassing human-level
#' performance on ImageNet classification` - He, K. et al. (2015), using a
#' uniform distribution.
#'
#' @param tensor an n-dimensional `torch.Tensor`
#' @param a the negative slope of the rectifier used after this layer (only used
#'  with `'leaky_relu'`)
#' @param mode either 'fan_in' (default) or 'fan_out'. Choosing 'fan_in' preserves
#'   the magnitude of the variance of the weights in the forward pass. Choosing
#'   'fan_out' preserves the magnitudes in the backwards pass.
#' @param nonlinearity the non-linear function. recommended to use only with 'relu'
#'   or 'leaky_relu' (default).
#'
#' @examples
#' w <- torch_empty(3, 5)
#' nn_init_kaiming_uniform_(w, mode = "fan_in", nonlinearity = "leaky_relu")
#' @export
nn_init_kaiming_uniform_ <- function(tensor, a = 0, mode = "fan_in", nonlinearity = "leaky_relu") {
  fan <- nn_init_calculate_correct_fan(tensor, mode)
  gain <- nn_init_calculate_gain(nonlinearity, a)
  std <- gain / sqrt(fan)
  bound <- sqrt(3) * std
  nn_init_no_grad_uniform(tensor, -bound, bound)
}

#' Kaiming normal initialization
#'
#' Fills the input `Tensor` with values according to the method
#' described in `Delving deep into rectifiers: Surpassing human-level
#' performance on ImageNet classification` - He, K. et al. (2015), using a
#' normal distribution.
#'
#' @inheritParams nn_init_kaiming_uniform_
#'
#' @examples
#' w <- torch_empty(3, 5)
#' nn_init_kaiming_normal_(w, mode = "fan_in", nonlinearity = "leaky_relu")
#' @export
nn_init_kaiming_normal_ <- function(tensor, a = 0, mode = "fan_in", nonlinearity = "leaky_relu") {
  fan <- nn_init_calculate_correct_fan(tensor, mode)
  gain <- nn_init_calculate_gain(nonlinearity, a)
  std <- gain / sqrt(fan)
  nn_init_no_grad_normal(tensor, 0, std)
}

#' Orthogonal initialization
#'
#' Fills the input `Tensor` with a (semi) orthogonal matrix, as
#' described in `Exact solutions to the nonlinear dynamics of learning in deep
#' linear neural networks` - Saxe, A. et al. (2013). The input tensor must have
#' at least 2 dimensions, and for tensors with more than 2 dimensions the
#' trailing dimensions are flattened.
#'
#' @param tensor an n-dimensional `Tensor`
#' @param gain optional scaling factor
#'
#' @examples
#' w <- torch_empty(3, 5)
#' nn_init_orthogonal_(w)
#' @export
nn_init_orthogonal_ <- function(tensor, gain = 1) {
  rows <- tensor$size(1)
  cols <- floor(tensor$numel() / rows)
  flattened <- torch_randn(rows, cols)

  if (rows < cols) {
    flattened$t_()
  }

  # Compute the qr factorization
  qr <- torch_qr(flattened)
  q <- qr[[1]]
  r <- qr[[2]]

  # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
  d <- torch_diag(r, 0)
  ph <- d$sign()
  q <- q * ph

  if (rows < cols) {
    q$t_()
  }

  with_no_grad({
    tensor$view_as(q)$copy_(q)
    tensor$mul_(gain)
  })

  tensor
}

#' Sparse initialization
#'
#' Fills the 2D input `Tensor` as a sparse matrix, where the
#' non-zero elements will be drawn from the normal distribution
#' as described in `Deep learning via
#' Hessian-free optimization` - Martens, J. (2010).
#'
#' @param tensor an n-dimensional `Tensor`
#' @param sparsity The fraction of elements in each column to be set to zero
#' @param std the standard deviation of the normal distribution used to generate
#'   the non-zero values
#'
#' @examples
#' \dontrun{
#' w <- torch_empty(3, 5)
#' nn_init_sparse_(w, sparsity = 0.1)
#' }
#' @export
nn_init_sparse_ <- function(tensor, sparsity, std = 0.01) {
  stop("not implemented")
}
