#' Absolute
#'
#' @section absolute(input, *, out=None) -> Tensor:
#'
#' Alias for [torch_abs()]
#' @inheritParams torch_abs
#'
#' @name torch_absolute
#'
#' @export
NULL


#' View_as_real
#'
#' @section view_as_real(input) -> Tensor :
#'
#' Returns a view of `input` as a real tensor. For an input complex tensor of
#' `size` \eqn{m1, m2, \dots, mi}, this function returns a new
#' real tensor of size \eqn{m1, m2, \dots, mi, 2}, where the last dimension of size 2
#' represents the real and imaginary components of complex numbers.
#' 
#' @section Warning:
#' [torch_view_as_real()] is only supported for tensors with `complex dtypes`.
#'
#' @param self (Tensor) the input tensor.
#'
#' @name torch_view_as_real
#'
#' @export
NULL

#' View_as_complex
#'
#' @section view_as_complex(input) -> Tensor :
#'
#' Returns a view of `input` as a complex tensor. For an input complex
#' tensor of `size` \eqn{m1, m2, \dots, mi, 2}, this function returns a
#' new complex tensor of `size` \eqn{m1, m2, \dots, mi} where the last
#' dimension of the input tensor is expected to represent the real and imaginary
#' components of complex numbers.
#' 
#' @section Warning:
#' [torch_view_as_complex] is only supported for tensors with
#' `torch_dtype` `torch_float64()` and `torch_float32()`.  The input is
#' expected to have the last dimension of `size` 2. In addition, the
#' tensor must have a `stride` of 1 for its last dimension. The strides of all
#' other dimensions must be even numbers.
#'
#' @param self (Tensor) the input tensor.
#'
#' @name torch_view_as_complex
#'
#' @export
NULL


#' Sgn
#'
#' @section sgn(input, *, out=None) -> Tensor :
#'
#' For complex tensors, this function returns a new tensor whose elemants have the same angle as that of the
#' elements of `input` and absolute value 1. For a non-complex tensor, this function
#' returns the signs of the elements of `input` (see [`torch_sign`]).
#' 
#' \eqn{\mbox{out}_{i} = 0}, if \eqn{|{\mbox{{input}}_i}| == 0}
#' \eqn{\mbox{out}_{i} = \frac{{\mbox{{input}}_i}}{|{\mbox{{input}}_i}|}}, otherwise
#'
#' @param self (Tensor) the input tensor.
#'
#' @name torch_sgn
#'
#' @export
NULL


#' Arccos
#'
#' @section arccos(input, *, out=None) -> Tensor :
#'
#' Alias for [torch_acos()].
#'
#' @inheritParams torch_acos
#' @name torch_arccos
#'
#' @export
NULL


#' Acosh
#'
#' @section acosh(input, *, out=None) -> Tensor :
#'
#' Returns a new tensor with the inverse hyperbolic cosine of the elements of `input`.
#' 
#' @note
#' The domain of the inverse hyperbolic cosine is `[1, inf)` and values outside this range
#' will be mapped to `NaN`, except for `+ INF` for which the output is mapped to `+ INF`.
#' 
#' \deqn{
#'     \mbox{out}_{i} = \cosh^{-1}(\mbox{input}_{i})
#' }
#'
#'
#' @param self (Tensor) the input tensor.
#'
#' @name torch_acosh
#'
#' @export
NULL


#' Arccosh
#'
#' @section arccosh(input, *, out=None) -> Tensor :
#'
#' Alias for [torch_acosh()].
#'
#' @inheritParams torch_amax
#' @name torch_arccosh
#'
#' @export
NULL


#' Asinh
#'
#' @section asinh(input, *, out=None) -> Tensor :
#'
#' Returns a new tensor with the inverse hyperbolic sine of the elements of `input`.
#' 
#' \deqn{
#'     \mbox{out}_{i} = \sinh^{-1}(\mbox{input}_{i})
#' }
#'
#' @param self (Tensor) the input tensor.
#'
#' @name torch_asinh
#'
#' @export
NULL

#' Arcsinh
#'
#' @section arcsinh(input, *, out=None) -> Tensor :
#'
#' Alias for [torch_asinh()].
#'
#' @inheritParams torch_asinh
#' @name torch_arcsinh
#'
#' @export
NULL


#' Atanh
#'
#' @section atanh(input, *, out=None) -> Tensor :
#'
#' Returns a new tensor with the inverse hyperbolic tangent of the elements of `input`.
#' 
#' @note
#' The domain of the inverse hyperbolic tangent is `(-1, 1)` and values outside this range
#' will be mapped to `NaN`, except for the values `1` and `-1` for which the output is
#' mapped to `+/-INF` respectively.
#' 
#' \deqn{
#'     \mbox{out}_{i} = \tanh^{-1}(\mbox{input}_{i})
#' }
#'
#' @param self (Tensor) the input tensor.
#'
#' @name torch_atanh
#'
#' @export
NULL

#' Arctanh
#'
#' @section arctanh(input, *, out=None) -> Tensor :
#'
#' Alias for [torch_atanh()].
#'
#' @inheritParams torch_atanh
#' @name torch_arctanh
#'
#' @export
NULL

#' Arcsin
#'
#' @section arcsin(input, *, out=None) -> Tensor :
#'
#' Alias for [torch_asin()].
#' 
#' @inheritParams torch_asin
#' @name torch_arcsin
#'
#' @export
NULL


#' Arctan
#'
#' @section arctan(input, *, out=None) -> Tensor :
#'
#' Alias for [torch_atan()].
#' 
#' @inheritParams torch_atan
#' @name torch_arctan
#'
#' @export
NULL


#' Atleast_1d
#'
#' Returns a 1-dimensional view of each input tensor with zero dimensions.
#' Input tensors with one or more dimensions are returned as-is.
#'
#' @param self (Tensor or list of Tensors) 
#'
#' @name torch_atleast_1d
#'
#' @export
NULL


#' Atleast_2d
#'
#' Returns a 2-dimensional view of each each input tensor with zero dimensions.
#' Input tensors with two or more dimensions are returned as-is.
#'
#' @param self (Tensor or list of Tensors) 
#'
#' @name torch_atleast_2d
#'
#' @export
NULL


#' Atleast_3d
#'
#' Returns a 3-dimensional view of each each input tensor with zero dimensions.
#' Input tensors with three or more dimensions are returned as-is.
#'
#' @param self (Tensor or list of Tensors) 
#'
#' @name torch_atleast_3d
#'
#' @export
NULL


#' Block_diag
#'
#' Create a block diagonal matrix from provided tensors.
#'
#' @param tensors (list of tensors) One or more tensors with 0, 1, or 2 
#'   dimensions.
#'
#' @name torch_block_diag
#'
#' @export
NULL

#' Unsafe_chunk
#'
#' @section unsafe_chunk(input, chunks, dim=0) -> List of Tensors :
#'
#' Works like [torch_chunk()] but without enforcing the autograd restrictions
#' on inplace modification of the outputs.
#' 
#' @inheritParams torch_chunk
#' 
#' @section Warning:
#' This function is safe to use as long as only the input, or only the outputs
#' are modified inplace after calling this function. It is user's
#' responsibility to ensure that is the case. If both the input and one or more
#' of the outputs are modified inplace, gradients computed by autograd will be
#' silently incorrect.
#'
#' @name torch_unsafe_chunk
#'
#' @export
NULL


#' Clip
#'
#' @section clip(input, min, max, *, out=None) -> Tensor :
#'
#' Alias for [torch_clamp()].
#'
#' @inheritParams torch_clamp
#' @name torch_clip
#'
#' @export
NULL


#' Complex
#'
#' @section complex(real, imag, *, out=None) -> Tensor :
#'
#' Constructs a complex tensor with its real part equal to `real` and its
#' imaginary part equal to `imag`.
#'
#' @param real (Tensor) The real part of the complex tensor. Must be float or double.
#' @param imag (Tensor) The imaginary part of the complex tensor. Must be same dtype 
#'   as `real`.
#'
#' @name torch_complex
#'
#' @export
NULL

#' Polar
#'
#' @section polar(abs, angle, *, out=None) -> Tensor :
#'
#' Constructs a complex tensor whose elements are Cartesian coordinates
#' corresponding to the polar coordinates with absolute value `abs` and angle
#' `angle`.
#' 
#' \deqn{
#'     \mbox{out} = \mbox{abs} \cdot \cos(\mbox{angle}) + \mbox{abs} \cdot \sin(\mbox{angle}) \cdot j
#' }
#'
#'
#' @param abs (Tensor) The absolute value the complex tensor. Must be float or 
#'   double.
#' @param angle (Tensor) The angle of the complex tensor. Must be same dtype as
#'   `abs`.
#'
#' @name torch_polar
#'
#' @export
NULL


#' Count_nonzero
#'
#' @section count_nonzero(input, dim=None) -> Tensor :
#'
#' Counts the number of non-zero values in the tensor `input` along the given `dim`.
#' If no dim is specified then all non-zeros in the tensor are counted.
#'
#' @param self (Tensor) the input tensor.
#' @param dim (int or tuple of ints, optional) Dim or tuple of dims along which 
#'   to count non-zeros.
#'
#' @name torch_count_nonzero
#'
#' @export
NULL


#' Divide
#'
#' @section divide(input, other, *, out=None) -> Tensor :
#'
#' Alias for [torch_div()].
#'
#' @inheritParams torch_div
#' @name torch_divide
#'
#' @export
NULL


#' Vdot
#'
#' @section vdot(input, other, *, out=None) -> Tensor :
#'
#' Computes the dot product (inner product) of two tensors. The vdot(a, b) function
#' handles complex numbers differently than dot(a, b). If the first argument is complex,
#' the complex conjugate of the first argument is used for the calculation of the dot product.
#' 
#' @note This function does not broadcast .
#'
#' @param self (Tensor) first tensor in the dot product. Its conjugate is used 
#'   if it's complex.
#' @param other (Tensor) second tensor in the dot product.
#'
#' @name torch_vdot
#'
#' @export
NULL


#' Exp2
#'
#' @section exp2(input, *, out=None) -> Tensor :
#'
#' Computes the base two exponential function of `input`.
#' 
#' \deqn{
#'     y_{i} = 2^{x_{i}}
#' }
#'
#'
#' @param self (Tensor) the input tensor.
#'
#' @name torch_exp2
#'
#' @export
NULL


#' Gcd
#'
#' @section gcd(input, other, *, out=None) -> Tensor :
#'
#' Computes the element-wise greatest common divisor (GCD) of `input` and `other`.
#' 
#' Both `input` and `other` must have integer types.
#' 
#' @note This defines \eqn{gcd(0, 0) = 0}.
#'
#'
#' @param self (Tensor) the input tensor.
#' @param other (Tensor) the second input tensor
#'
#' @name torch_gcd
#'
#' @export
NULL


#' Lcm
#'
#' @section lcm(input, other, *, out=None) -> Tensor :
#'
#' Computes the element-wise least common multiple (LCM) of `input` and `other`.
#' 
#' Both `input` and `other` must have integer types.
#' 
#' @note This defines \eqn{lcm(0, 0) = 0} and \eqn{lcm(0, a) = 0}.
#'
#' @param self (Tensor) the input tensor.
#' @param other (Tensor) the second input tensor
#'
#' @name torch_lcm
#'
#' @export
NULL


#' Kaiser_window
#'
#' @section kaiser_window(window_length, periodic=TRUE, beta=12.0, *, dtype=None, layout=torch.strided, device=None, requires_grad=FALSE) -> Tensor :
#'
#' Computes the Kaiser window with window length `window_length` and shape parameter `beta`.
#' 
#' Let I_0 be the zeroth order modified Bessel function of the first kind (see [torch_i0()]) and
#' `N = L - 1` if `periodic` is FALSE and `L` if `periodic` is TRUE,
#' where `L` is the `window_length`. This function computes:
#' 
#' \deqn{
#'     out_i = I_0 \left( \beta \sqrt{1 - \left( {\frac{i - N/2}{N/2}} \right) ^2 } \right) / I_0( \beta )
#' }
#' 
#' Calling `torch_kaiser_window(L, B, periodic=TRUE)` is equivalent to calling
#' `torch_kaiser_window(L + 1, B, periodic=FALSE)[:-1])`.
#' The `periodic` argument is intended as a helpful shorthand
#' to produce a periodic window as input to functions like [torch_stft()].
#' 
#' @note
#' If `window_length` is one, then the returned window is a single element 
#' tensor containing a one.
#'
#' @param window_length (int) length of the window.
#' @param periodic (bool, optional) If TRUE, returns a periodic window suitable for use in spectral analysis.        If FALSE, returns a symmetric window suitable for use in filter design.
#' @param beta (float, optional) shape parameter for the window.
#' @inheritParams torch_tensor
#'
#' @name torch_kaiser_window
#'
#' @export
NULL


#' Isclose
#'
#' @section isclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=FALSE) -> Tensor :
#'
#' Returns a new tensor with boolean elements representing if each element of
#' `input` is "close" to the corresponding element of `other`.
#' Closeness is defined as:
#' 
#' \deqn{
#'     \vert \mbox{input} - \mbox{other} \vert \leq \mbox{atol} + \mbox{rtol} \times \vert \mbox{other} \vert
#' }
#' 
#' where `input` and `other` are finite. Where `input`
#' and/or `other` are nonfinite they are close if and only if
#' they are equal, with NaNs being considered equal to each other when
#' `equal_nan` is TRUE.
#'
#'
#' @param self (Tensor) first tensor to compare
#' @param other (Tensor) second tensor to compare
#' @param atol (float, optional) absolute tolerance. Default: 1e-08
#' @param rtol (float, optional) relative tolerance. Default: 1e-05
#' @param equal_nan (bool, optional) if `TRUE`, then two `NaN` s will be 
#'   considered equal. Default: `FALSE`
#'
#' @name torch_isclose
#'
#' @export
NULL


#' Isreal
#'
#' @section isreal(input) -> Tensor :
#'
#' Returns a new tensor with boolean elements representing if each element of `input` is real-valued or not.
#' All real-valued types are considered real. Complex values are considered real when their imaginary part is 0.
#'
#'
#' @param self (Tensor) the input tensor.
#'
#' @name torch_isreal
#'
#' @export
NULL


#' Is_nonzero
#'
#' @section is_nonzero(input) -> (bool) :
#'
#' Returns TRUE if the `input` is a single element tensor which is not equal to zero
#' after type conversions.
#' i.e. not equal to `torch_tensor(c(0))` or `torch_tensor(c(0))` or
#' `torch_tensor(c(FALSE))`.
#' Throws a `RuntimeError` if `torch_numel() != 1` (even in case
#' of sparse tensors).
#'
#' @param self (Tensor) the input tensor.
#'
#' @name torch_is_nonzero
#'
#' @export
NULL


#' Logaddexp
#'
#' @section logaddexp(input, other, *, out=None) -> Tensor :
#'
#' Logarithm of the sum of exponentiations of the inputs.
#' 
#' Calculates pointwise \eqn{\log\left(e^x + e^y\right)}. This function is useful
#' in statistics where the calculated probabilities of events may be so small as to
#' exceed the range of normal floating point numbers. In such cases the logarithm
#' of the calculated probability is stored. This function allows adding
#' probabilities stored in such a fashion.
#' 
#' This op should be disambiguated with [torch_logsumexp()] which performs a
#' reduction on a single tensor.
#'
#'
#' @param self (Tensor) the input tensor.
#' @param other (Tensor) the second input tensor
#'
#' @name torch_logaddexp
#'
#' @export
NULL


#' Logaddexp2
#'
#' @section logaddexp2(input, other, *, out=None) -> Tensor :
#'
#' Logarithm of the sum of exponentiations of the inputs in base-2.
#' 
#' Calculates pointwise \eqn{\log_2\left(2^x + 2^y\right)}. See
#' [torch_logaddexp()] for more details.
#'
#'
#' @param self (Tensor) the input tensor.
#' @param other (Tensor) the second input tensor
#'
#' @name torch_logaddexp2
#'
#' @export
NULL


#' Logcumsumexp
#'
#' @section logcumsumexp(input, dim, *, out=None) -> Tensor :
#'
#' Returns the logarithm of the cumulative summation of the exponentiation of
#' elements of `input` in the dimension `dim`.
#' 
#' For summation index \eqn{j} given by `dim` and other indices \eqn{i}, the result is
#' 
#' \deqn{
#'         \mbox{logcumsumexp}(x)_{ij} = \log \sum\limits_{j=0}^{i} \exp(x_{ij})
#' }
#'
#'
#' @param self (Tensor) the input tensor.
#' @param dim (int) the dimension to do the operation over
#'
#' @name torch_logcumsumexp
#'
#' @export
NULL


#' Matrix_exp
#'
#' @section matrix_power(input) -> Tensor :
#'
#' Returns the matrix exponential. Supports batched input.
#' For a matrix `A`, the matrix exponential is defined as
#' 
#' \deqn{
#'     \exp^A = \sum_{k=0}^\infty A^k / k!.
#' }
#' 
#' The implementation is based on:
#' Bader, P.; Blanes, S.; Casas, F.
#' Computing the Matrix Exponential with an Optimized Taylor Polynomial Approximation.
#' Mathematics 2019, 7, 1174.
#'
#'
#' @param self (Tensor) the input tensor.
#'
#' @name torch_matrix_exp
#'
#' @export
NULL


#' Amax
#'
#' @section amax(input, dim, keepdim=FALSE, *, out=None) -> Tensor :
#'
#' Returns the maximum value of each slice of the `input` tensor in the given
#' dimension(s) `dim`.
#' 
#' @note
#' The difference between `max`/`min` and `amax`/`amin` is:
#' - `amax`/`amin` supports reducing on multiple dimensions,
#' - `amax`/`amin` does not return indices,
#' - `amax`/`amin` evenly distributes gradient between equal values,
#'    while `max(dim)`/`min(dim)` propagates gradient only to a single
#'    index in the source tensor.
#' 
#' If `keepdim is `TRUE`, the output tensors are of the same size
#' as `input` except in the dimension(s) `dim` where they are of size 1.
#' Otherwise, `dim`s are squeezed (see [torch_squeeze()]), resulting
#' in the output tensors having fewer dimension than `input`.
#'
#'
#' @param self (Tensor) the input tensor.
#' @param dim (int or tuple of ints) the dimension or dimensions to reduce.
#' @param keepdim (bool) whether the output tensor has `dim` retained or not.
#'
#' @name torch_amax
#'
#' @export
NULL


#' Amin
#'
#' @section amin(input, dim, keepdim=FALSE, *, out=None) -> Tensor :
#'
#' Returns the minimum value of each slice of the `input` tensor in the given
#' dimension(s) `dim`.
#' 
#' @note
#' The difference between `max`/`min` and `amax`/`amin` is:
#' - `amax`/`amin` supports reducing on multiple dimensions,
#' - `amax`/`amin` does not return indices,
#' - `amax`/`amin` evenly distributes gradient between equal values,
#'    while `max(dim)`/`min(dim)` propagates gradient only to a single
#'    index in the source tensor.
#' 
#' If `keepdim` is `TRUE`, the output tensors are of the same size as
#' `input` except in the dimension(s) `dim` where they are of size 1.
#' Otherwise, `dim`s are squeezed (see [torch_squeeze()]), resulting in
#' the output tensors having fewer dimensions than `input`.
#'
#'
#' @param self (Tensor) the input tensor.
#' @param dim (int or tuple of ints) the dimension or dimensions to reduce.
#' @param keepdim (bool) whether the output tensor has `dim` retained or not.
#'
#' @name torch_amin
#'
#' @export
NULL


#' Multiply
#'
#' @section multiply(input, other, *, out=None) :
#'
#' Alias for [torch_mul()].
#'
#' @inheritParams torch_mul
#' @name torch_multiply
#'
#' @export
NULL


#' Movedim
#'
#' @section movedim(input, source, destination) -> Tensor :
#'
#' Moves the dimension(s) of `input` at the position(s) in `source`
#' to the position(s) in `destination`.
#' 
#' Other dimensions of `input` that are not explicitly moved remain in
#' their original order and appear at the positions not specified in `destination`.
#'
#' @param self (Tensor) the input tensor.
#' @param source (int or tuple of ints) Original positions of the dims to move. These must be unique.
#' @param destination (int or tuple of ints) Destination positions for each of the original dims. These must also be unique.
#'
#' @name torch_movedim
#'
#' @export
NULL


#' Channel_shuffle
#'
#' @section Divide the channels in a tensor of shape :math:`(*, C , H, W)` :
#'
#' Divide the channels in a tensor of shape \eqn{(*, C , H, W)}
#' into g groups and rearrange them as \eqn{(*, C \frac g, g, H, W)},
#' while keeping the original tensor shape.
#'
#' @param self (Tensor) the input tensor
#' @param groups (int) number of groups to divide channels in and rearrange.
#'
#' @name torch_channel_shuffle
#'
#' @export
NULL


#' Rad2deg
#'
#' @section rad2deg(input, *, out=None) -> Tensor :
#'
#' Returns a new tensor with each of the elements of `input`
#' converted from angles in radians to degrees.
#'
#'
#' @param self (Tensor) the input tensor.
#'
#' @name torch_rad2deg
#'
#' @export
NULL


#' Deg2rad
#'
#' @section deg2rad(input, *, out=None) -> Tensor :
#'
#' Returns a new tensor with each of the elements of `input`
#' converted from angles in degrees to radians.
#'
#'
#' @param self (Tensor) the input tensor.
#'
#' @name torch_deg2rad
#'
#' @export
NULL


#' Negative
#'
#' @section negative(input, *, out=None) -> Tensor :
#'
#' Alias for [torch_neg()]
#' 
#' @inheritParams torch_neg
#' @name torch_negative
#'
#' @export
NULL


#' Logit
#'
#' @section logit(input, eps=None, *, out=None) -> Tensor :
#'
#' Returns a new tensor with the logit of the elements of `input`.
#' `input` is clamped to `[eps, 1 - eps]` when eps is not None.
#' When eps is None and `input` < 0 or `input` > 1, the function will yields NaN.
#' 
#' \deqn{
#'     y_{i} = \ln(\frac{z_{i}}{1 - z_{i}}) \\
#'     z_{i} = \begin{array}{ll}
#'         x_{i} & \mbox{if eps is None} \\
#'         \mbox{eps} & \mbox{if } x_{i} < \mbox{eps} \\
#'         x_{i} & \mbox{if } \mbox{eps} \leq x_{i} \leq 1 - \mbox{eps} \\
#'         1 - \mbox{eps} & \mbox{if } x_{i} > 1 - \mbox{eps}
#'     \end{array}
#' }
#'
#'
#' @param self (Tensor) the input tensor.
#' @param eps (float, optional) the epsilon for input clamp bound. Default: `None`
#'
#' @name torch_logit
#'
#' @export
NULL


#' Unsafe_split
#'
#' @section unsafe_split(tensor, split_size_or_sections, dim=0) -> List of Tensors :
#'
#' Works like [torch_split()] but without enforcing the autograd restrictions
#' on inplace modification of the outputs.
#' 
#' @section Warning:
#' This function is safe to use as long as only the input, or only the outputs
#' are modified inplace after calling this function. It is user's
#' responsibility to ensure that is the case. If both the input and one or more
#' of the outputs are modified inplace, gradients computed by autograd will be
#' silently incorrect.
#' 
#' @inheritParams torch_split
#' @name torch_unsafe_split
#'
#' @export
NULL


#' Hstack
#'
#' @section hstack(tensors, *, out=None) -> Tensor :
#'
#' Stack tensors in sequence horizontally (column wise).
#' 
#' This is equivalent to concatenation along the first axis for 1-D tensors, and 
#' along the second axis for all other tensors.
#'
#' @param tensors (sequence of Tensors) sequence of tensors to concatenate
#'
#' @name torch_hstack
#'
#' @export
NULL


#' Vstack
#'
#' @section vstack(tensors, *, out=None) -> Tensor :
#'
#' Stack tensors in sequence vertically (row wise).
#' 
#' This is equivalent to concatenation along the first axis after all 1-D tensors 
#' have been reshaped by [torch_atleast_2d()].
#'
#' @param tensors (sequence of Tensors) sequence of tensors to concatenate
#'
#' @name torch_vstack
#'
#' @export
NULL


#' Dstack
#'
#' @section dstack(tensors, *, out=None) -> Tensor :
#'
#' Stack tensors in sequence depthwise (along third axis).
#' 
#' This is equivalent to concatenation along the third axis after 1-D and 2-D 
#' tensors have been reshaped by [torch_atleast_3d()].
#'
#' @param tensors (sequence of Tensors) sequence of tensors to concatenate
#'
#' @name torch_dstack
#'
#' @export
NULL


#' Istft
#'
#' Inverse short time Fourier Transform. This is expected to be the inverse of [torch_stft()].
#' 
#' It has the same parameters (+ additional optional parameter of `length`) and it should return the
#' least squares estimation of the original signal. The algorithm will check using the NOLA 
#' condition (nonzero overlap).
#' 
#' Important consideration in the parameters `window` and `center` so that the envelop
#' created by the summation of all the windows is never zero at certain point in time. Specifically,
#' \eqn{\sum_{t=-\infty}^{\infty} |w|^2\[n-t\times hop_length\] \neq 0}.
#' 
#' Since [torch_stft()] discards elements at the end of the signal if they do not fit in a frame,
#' `istft` may return a shorter signal than the original signal (can occur if `center` is FALSE
#' since the signal isn't padded).
#' 
#' If `center` is `TRUE`, then there will be padding e.g. `'constant'`, `'reflect'`, etc.
#' Left padding can be trimmed off exactly because they can be calculated but right 
#' padding cannot be calculated without additional information.
#' 
#' Example: Suppose the last window is:
#' `[c(17, 18, 0, 0, 0)` vs `c(18, 0, 0, 0, 0)`
#' 
#' The `n_fft`, `hop_length`, `win_length` are all the same which prevents the calculation
#' of right padding. These additional values could be zeros or a reflection of the signal so providing
#' `length` could be useful. If `length` is `None` then padding will be aggressively removed
#' (some loss of signal).
#' 
#' D. W. Griffin and J. S. Lim, "Signal estimation from modified short-time Fourier transform,"
#' IEEE Trans. ASSP, vol.32, no.2, pp.236-243, Apr. 1984.
#'
#' @param self (Tensor) The input tensor. Expected to be output of [torch_stft()], 
#'   can either be complex (`channel`, `fft_size`, `n_frame`), or real 
#'   (`channel`, `fft_size`, `n_frame`, 2) where the `channel` dimension is 
#'   optional.
#' @param n_fft (int) Size of Fourier transform
#' @param hop_length (Optional[int]) The distance between neighboring sliding window frames.
#'   (Default: `n_fft %% 4`)
#' @param win_length (Optional[int]) The size of window frame and STFT filter. 
#'   (Default: `n_fft`)
#' @param window (Optional(torch.Tensor)) The optional window function.
#'   (Default: `torch_ones(win_length)`)
#' @param center (bool) Whether `input` was padded on both sides so that the 
#'   \eqn{t}-th frame is centered at time \eqn{t \times \mbox{hop\_length}}.
#'   (Default: `TRUE`)
#' @param normalized (bool) Whether the STFT was normalized. (Default: `FALSE`)
#' @param onesided (Optional[bool]) Whether the STFT was onesided. 
#'   (Default: `TRUE` if `n_fft != fft_size` in the input size)
#' @param length (Optional[int]) The amount to trim the signal by (i.e. the 
#'   original signal length). (Default: whole signal)
#' @param return_complex (Optional(bool)) Whether the output should be complex, 
#'   or if the input should be assumed to derive from a real signal and window. 
#'   Note that this is incompatible with `onesided=TRUE`. (Default: `FALSE`)
#'
#' @name torch_istft
#'
#' @export
NULL


#' Nansum
#'
#' @section nansum(input, *, dtype=None) -> Tensor :
#'
#' Returns the sum of all elements, treating Not a Numbers (NaNs) as zero.
#'
#' @section nansum(input, dim, keepdim=FALSE, *, dtype=None) -> Tensor :
#'
#' Returns the sum of each row of the `input` tensor in the given
#' dimension `dim`, treating Not a Numbers (NaNs) as zero.
#' If `dim` is a list of dimensions, reduce over all of them.
#' 
#' If `keepdim` is `TRUE`, the output tensor is of the same size
#' as `input` except in the dimension(s) `dim` where it is of size 1.
#' Otherwise, `dim` is squeezed (see [`torch_squeeze`]), resulting in the
#' output tensor having 1 (or `len(dim)`) fewer dimension(s).
#'
#' @param self (Tensor) the input tensor.
#' @param dim (int or tuple of ints) the dimension or dimensions to reduce.
#' @param keepdim (bool) whether the output tensor has `dim` retained or not.
#' @param dtype the desired data type of returned tensor. If specified, the 
#'   input tensor is casted to dtype before the operation is performed. This is 
#'   useful for preventing data type overflows. Default: `NULL`.
#'
#' @name torch_nansum
#'
#' @export
NULL

#' Fliplr
#'
#' @section fliplr(input) -> Tensor :
#'
#' Flip array in the left/right direction, returning a new tensor.
#' 
#' Flip the entries in each row in the left/right direction.
#' Columns are preserved, but appear in a different order than before.
#' 
#' @note
#' Equivalent to `input[,-1]`. Requires the array to be at least 2-D.
#'
#'
#' @param self (Tensor) Must be at least 2-dimensional.
#'
#' @name torch_fliplr
#'
#' @export
NULL


#' Flipud
#'
#' @section flipud(input) -> Tensor :
#'
#' Flip array in the up/down direction, returning a new tensor.
#' 
#' Flip the entries in each column in the up/down direction.
#' Rows are preserved, but appear in a different order than before.
#' 
#' @note
#' Equivalent to `input[-1,]`. Requires the array to be at least 1-D.
#'
#'
#' @param self (Tensor) Must be at least 1-dimensional.
#'
#' @name torch_flipud
#'
#' @export
NULL


#' Fix
#'
#' @section fix(input, *, out=None) -> Tensor :
#'
#' Alias for [torch_trunc()]
#'
#' @inheritParams torch_trunc
#' @name torch_fix
#'
#' @export
NULL


#' Vander
#'
#' @section vander(x, N=None, increasing=FALSE) -> Tensor :
#'
#' Generates a Vandermonde matrix.
#' 
#' The columns of the output matrix are elementwise powers of the input vector 
#' \eqn{x^{(N-1)}, x^{(N-2)}, ..., x^0}.
#' If increasing is TRUE, the order of the columns is reversed 
#' \eqn{x^0, x^1, ..., x^{(N-1)}}. Such a
#' matrix with a geometric progression in each row is 
#' named for Alexandre-Theophile Vandermonde.
#'
#'
#' @param x (Tensor) 1-D input tensor.
#' @param N (int, optional) Number of columns in the output. If N is not specified,        
#'   a square array is returned \eqn{(N = len(x))}.
#' @param increasing (bool, optional) Order of the powers of the columns. If TRUE,        
#'   the powers increase from left to right, if FALSE (the default) they are reversed.
#'
#' @name torch_vander
#'
#' @export
NULL


#' Clone
#'
#' @section clone(input, *, memory_format=torch.preserve_format) -> Tensor :
#'
#' Returns a copy of `input`.
#' 
#' @note
#' 
#' This function is differentiable, so gradients will flow back from the
#' result of this operation to `input`. To create a tensor without an
#' autograd relationship to `input` see `Tensor$detach`.
#'
#' @param self (Tensor) the input tensor.
#' @param memory_format a torch memory format. see [torch_preserve_format()].
#' 
#' @name torch_clone
#'
#' @export
NULL


#' Sub
#'
#' @section sub(input, other, *, alpha=1, out=None) -> Tensor :
#'
#' Subtracts `other`, scaled by `alpha`, from `input`.
#' 
#' \deqn{
#'     \mbox{{out}}_i = \mbox{{input}}_i - \mbox{{alpha}} \times \mbox{{other}}_i
#' }
#' 
#' Supports broadcasting to a common shape ,
#' type promotion , and integer, float, and complex inputs.
#'
#' @param self (Tensor) the input tensor.
#' @param other (Tensor or Scalar) the tensor or scalar to subtract from `input`
#' @param alpha the scalar multiplier for other
#' 
#' @name torch_sub
#'
#' @export
NULL


#' Subtract
#'
#' @section subtract(input, other, *, alpha=1, out=None) -> Tensor :
#'
#' Alias for [torch_sub()].
#'
#' @inheritParams torch_sub
#' @name torch_subtract
#'
#' @export
NULL


#' Heaviside
#'
#' @section heaviside(input, values, *, out=None) -> Tensor :
#'
#' Computes the Heaviside step function for each element in `input`.
#' The Heaviside step function is defined as:
#' 
#' \deqn{
#' \mbox{{heaviside}}(input, values) = \begin{array}{ll}
#'  0, & \mbox{if input < 0}\\
#'  values, & \mbox{if input == 0}\\
#'  1, & \mbox{if input > 0}
#'  \end{array}
#' }
#'
#'
#' @param self (Tensor) the input tensor.
#' @param values (Tensor) The values to use where `input` is zero.
#'
#' @name torch_heaviside
#'
#' @export
NULL


#' Dequantize
#'
#' @section dequantize(tensor) -> Tensor :
#'
#' Returns an fp32 Tensor by dequantizing a quantized Tensor
#'
#' @section dequantize(tensors) -> sequence of Tensors :
#'
#' Given a list of quantized Tensors, dequantize them and return a list of fp32 Tensors
#'
#' @param tensor (Tensor) A quantized Tensor or a list oof quantized tensors
#'
#' @name torch_dequantize
#'
#' @export
NULL


#' Not_equal
#'
#' @section not_equal(input, other, *, out=None) -> Tensor :
#'
#' Alias for [torch_ne()].
#'
#' @inheritParams torch_ne
#' @name torch_not_equal
#'
#' @export
NULL


#' Greater_equal
#'
#' @section greater_equal(input, other, *, out=None) -> Tensor :
#'
#' Alias for [torch_ge()].
#'
#' @inheritParams torch_ge
#' @name torch_greater_equal
#'
#' @export
NULL


#' Less_equal
#'
#' @section less_equal(input, other, *, out=None) -> Tensor :
#'
#' Alias for [torch_le()].
#'
#' @inheritParams torch_le
#' @name torch_less_equal
#'
#' @export
NULL


#' Greater
#'
#' @section greater(input, other, *, out=None) -> Tensor :
#'
#' Alias for [torch_gt()].
#'
#' @inheritParams torch_gt
#' @name torch_greater
#'
#' @export
NULL


#' Less
#'
#' @section less(input, other, *, out=None) -> Tensor :
#'
#' Alias for [torch_lt()].
#'
#' @inheritParams torch_lt
#' @name torch_less
#'
#' @export
NULL


#' I0
#'
#' @section i0(input, *, out=None) -> Tensor :
#'
#' Computes the zeroth order modified Bessel function of the first kind for each element of `input`.
#' 
#' \deqn{
#' \mbox{out}_{i} = I_0(\mbox{input}_{i}) = \sum_{k=0}^{\infty} \frac{(\mbox{input}_{i}^2/4)^k}{(k!)^2}
#' }
#'
#' @param self (Tensor) the input tensor
#'
#' @name torch_i0
#'
#' @export
NULL


#' Signbit
#'
#' @section signbit(input, *, out=None) -> Tensor :
#'
#' Tests if each element of `input` has its sign bit set (is less than zero) or not.
#'
#' @param self (Tensor) the input tensor.
#'
#' @name torch_signbit
#'
#' @export
NULL


#' Hypot
#'
#' @section hypot(input, other, *, out=None) -> Tensor :
#'
#' Given the legs of a right triangle, return its hypotenuse.
#' 
#' \deqn{
#' \mbox{out}_{i} = \sqrt{\mbox{input}_{i}^{2} + \mbox{other}_{i}^{2}}
#' }
#' 
#' The shapes of `input` and `other` must be
#' broadcastable .
#'
#'
#' @param self (Tensor) the first input tensor
#' @param other (Tensor) the second input tensor
#'
#' @name torch_hypot
#'
#' @export
NULL


#' Nextafter
#'
#' @section nextafter(input, other, *, out=None) -> Tensor :
#'
#' Return the next floating-point value after `input` towards `other`, elementwise.
#' 
#' The shapes of `input` and `other` must be
#' broadcastable .
#'
#'
#' @param self (Tensor) the first input tensor
#' @param other (Tensor) the second input tensor
#'
#' @name torch_nextafter
#'
#' @export
NULL


#' Maximum
#'
#' @section maximum(input, other, *, out=None) -> Tensor :
#'
#' Computes the element-wise maximum of `input` and `other`.
#' 
#' @note
#' If one of the elements being compared is a NaN, then that element is returned.
#' [torch_maximum()] is not supported for tensors with complex dtypes.
#'
#'
#' @param self (Tensor) the input tensor.
#' @param other (Tensor) the second input tensor
#'
#' @name torch_maximum
#'
#' @export
NULL


#' Minimum
#'
#' @section minimum(input, other, *, out=None) -> Tensor :
#'
#' Computes the element-wise minimum of `input` and `other`.
#' 
#' @note
#' If one of the elements being compared is a NaN, then that element is returned.
#' [torch_minimum()] is not supported for tensors with complex dtypes.
#'
#'
#' @param self (Tensor) the input tensor.
#' @param other (Tensor) the second input tensor
#'
#' @name torch_minimum
#'
#' @export
NULL


#' Quantile
#'
#' @section quantile(input, q) -> Tensor :
#'
#' Returns the q-th quantiles of all elements in the `input` tensor, doing a linear
#' interpolation when the q-th quantile lies between two data points.
#'
#' @section quantile(input, q, dim=None, keepdim=FALSE, *, out=None) -> Tensor :
#'
#' Returns the q-th quantiles of each row of the `input` tensor along the dimension
#' `dim`, doing a linear interpolation when the q-th quantile lies between two
#' data points. By default, `dim` is `None` resulting in the `input` tensor
#' being flattened before computation.
#' 
#' If `keepdim` is `TRUE`, the output dimensions are of the same size as `input`
#' except in the dimensions being reduced (`dim` or all if `dim` is `NULL`) where they
#' have size 1. Otherwise, the dimensions being reduced are squeezed (see [`torch_squeeze`]).
#' If `q` is a 1D tensor, an extra dimension is prepended to the output tensor with the same
#' size as `q` which represents the quantiles.
#'
#'
#' @param self (Tensor) the input tensor.
#' @param q (float or Tensor) a scalar or 1D tensor of quantile values in the range `[0, 1]`
#' @param dim (int) the dimension to reduce.
#' @param keepdim (bool) whether the output tensor has `dim` retained or not.
#'
#' @name torch_quantile
#'
#' @export
NULL


#' Nanquantile
#'
#' @section nanquantile(input, q, dim=None, keepdim=FALSE, *, out=None) -> Tensor :
#'
#' This is a variant of [torch_quantile()] that "ignores" `NaN` values,
#' computing the quantiles `q` as if `NaN` values in `input` did
#' not exist. If all values in a reduced row are `NaN` then the quantiles for
#' that reduction will be `NaN`. See the documentation for [torch_quantile()].
#'
#'
#' @param self (Tensor) the input tensor.
#' @param q (float or Tensor) a scalar or 1D tensor of quantile values in the range `[0, 1]`
#' @param dim (int) the dimension to reduce.
#' @param keepdim (bool) whether the output tensor has `dim` retained or not.
#'
#' @name torch_nanquantile
#'
#' @export
NULL


#' Bucketize
#'
#' @section bucketize(input, boundaries, *, out_int32=FALSE, right=FALSE, out=None) -> Tensor :
#'
#' Returns the indices of the buckets to which each value in the `input` belongs, where the
#' boundaries of the buckets are set by `boundaries`. Return a new tensor with the same size
#' as `input`. If `right` is FALSE (default), then the left boundary is closed. 
#'
#' @param self (Tensor or Scalar) N-D tensor or a Scalar containing the search value(s).
#' @param boundaries (Tensor) 1-D tensor, must contain a monotonically increasing sequence.
#' @param out_int32 (bool, optional) – indicate the output data type. [torch_int32()] 
#'  if True, [torch_int64()] otherwise. Default value is FALSE, i.e. default output 
#'  data type is [torch_int64()].
#' @param right (bool, optional) – if False, return the first suitable location 
#'  that is found. If True, return the last such index. If no suitable index found, 
#'  return 0 for non-numerical value (eg. nan, inf) or the size of boundaries 
#'  (one pass the last index). In other words, if False, gets the lower bound index 
#'  for each value in input from boundaries. If True, gets the upper bound index 
#'  instead. Default value is False.
#'
#' @name torch_bucketize
#'
#' @export
NULL


#' Searchsorted
#'
#' @section searchsorted(sorted_sequence, values, *, out_int32=FALSE, right=FALSE, out=None) -> Tensor :
#'
#' Find the indices from the *innermost* dimension of `sorted_sequence` such that, if the
#' corresponding values in `values` were inserted before the indices, the order of the
#' corresponding *innermost* dimension within `sorted_sequence` would be preserved.
#' Return a new tensor with the same size as `values`. If `right` is FALSE (default),
#' then the left boundary of `sorted_sequence` is closed.
#'
#' @param sorted_sequence (Tensor) N-D or 1-D tensor, containing monotonically increasing 
#'   sequence on the *innermost* dimension.
#' @param values (Tensor or Scalar) N-D tensor or a Scalar containing the search value(s).
#' 
#' @inheritParams torch_bucketize
#' @name torch_searchsorted
#'
#' @export
NULL


#' Isposinf
#'
#' @section isposinf(input, *, out=None) -> Tensor :
#'
#' Tests if each element of `input` is positive infinity or not.
#'
#' @param self (Tensor) the input tensor.
#'
#' @name torch_isposinf
#'
#' @export
NULL


#' Isneginf
#'
#' @section isneginf(input, *, out=None) -> Tensor :
#'
#' Tests if each element of `input` is negative infinity or not.
#'
#' @param self (Tensor) the input tensor.
#'
#' @name torch_isneginf
#'
#' @export
NULL


#' Outer
#'
#' @section outer(input, vec2, *, out=None) -> Tensor :
#'
#' Outer product of `input` and `vec2`.
#' If `input` is a vector of size \eqn{n} and `vec2` is a vector of
#' size \eqn{m}, then `out` must be a matrix of size \eqn{(n \times m)}.
#' 
#' @note This function does not broadcast.
#'
#' @param self (Tensor) 1-D input vector
#' @param vec2 (Tensor) 1-D input vector
#'
#' @name torch_outer
#'
#' @export
NULL
