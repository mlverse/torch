#' Abs
#'
#' Computes the element-wise absolute value of the given `input` tensor.
#' 
#' \deqn{
#'     \text{out}_{i} = |\text{input}_{i}|
#' }
#'
#' @section Signatures:
#' 
#' abs(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' torch_abs(torch_tensor([-1, -2, 3]))
#' }
#'
#' @name torch_abs
#'
#' @export
NULL


#' Angle
#'
#' Computes the element-wise angle (in radians) of the given `input` tensor.
#' 
#' \deqn{
#'     \text{out}_{i} = angle(\text{input}_{i})
#' }
#'
#' @section Signatures:
#' 
#' angle(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' torch_angle(torch_tensor([-1 + 1j, -2 + 2j, 3 - 3j]))*180/3.14159
#' }
#'
#' @name torch_angle
#'
#' @export
NULL


#' Real
#'
#' Computes the element-wise real value of the given `input` tensor.
#' 
#' \deqn{
#'     \text{out}_{i} = real(\text{input}_{i})
#' }
#'
#' @section Signatures:
#' 
#' real(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' torch_real(torch_tensor([-1 + 1j, -2 + 2j, 3 - 3j]))
#' }
#'
#' @name torch_real
#'
#' @export
NULL


#' Imag
#'
#' Computes the element-wise imag value of the given `input` tensor.
#' 
#' \deqn{
#'     \text{out}_{i} = imag(\text{input}_{i})
#' }
#'
#' @section Signatures:
#' 
#' imag(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' torch_imag(torch_tensor([-1 + 1j, -2 + 2j, 3 - 3j]))
#' }
#'
#' @name torch_imag
#'
#' @export
NULL


#' Conj
#'
#' Computes the element-wise conjugate of the given `input` tensor.
#' 
#' \deqn{
#'     \text{out}_{i} = conj(\text{input}_{i})
#' }
#'
#' @section Signatures:
#' 
#' conj(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' torch_conj(torch_tensor([-1 + 1j, -2 + 2j, 3 - 3j]))
#' }
#'
#' @name torch_conj
#'
#' @export
NULL


#' Acos
#'
#' Returns a new tensor with the arccosine  of the elements of `input`.
#' 
#' \deqn{
#'     \text{out}_{i} = \cos^{-1}(\text{input}_{i})
#' }
#'
#' @section Signatures:
#' 
#' acos(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4)
#' a
#' torch_acos(a)
#' }
#'
#' @name torch_acos
#'
#' @export
NULL


#' Avg_pool1d
#'
#' Applies a 1D average pooling over an input signal composed of several
#' input planes.
#' 
#' See :class:`~torch.nn.AvgPool1d` for details and output shape.
#'
#' @section Signatures:
#' 
#' avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True) -> Tensor
#'
#'
#' @param input NA input tensor of shape \eqn{(\text{minibatch} , \text{in\_channels} , iW)}
#' @param kernel_size NA the size of the window. Can be a single number or a      tuple `(kW,)`
#' @param stride NA the stride of the window. Can be a single number or a tuple      `(sW,)`. Default: :attr:`kernel_size`
#' @param padding NA implicit zero paddings on both sides of the input. Can be a      single number or a tuple `(padW,)`. Default: 0
#' @param ceil_mode NA when True, will use `ceil` instead of `floor` to compute the        output shape. Default: ``False``
#' @param count_include_pad NA when True, will include the zero-padding in the        averaging calculation. Default: ``True``
#'
#' 
#'
#' @name torch_avg_pool1d
#'
#' @export
NULL


#' Adaptive_avg_pool1d
#'
#' Applies a 1D adaptive average pooling over an input signal composed of
#' several input planes.
#' 
#' See :class:`~torch.nn.AdaptiveAvgPool1d` for details and output shape.
#'
#' @section Signatures:
#' 
#' adaptive_avg_pool1d(input, output_size) -> Tensor
#'
#'
#' @param output_size NA the target output size (single integer)
#'
#' 
#'
#' @name torch_adaptive_avg_pool1d
#'
#' @export
NULL


#' Add
#'
#' add(input, other, out=None)
#' 
#' Adds the scalar `other` to each element of the input `input`
#' and returns a new resulting tensor.
#' 
#' \deqn{
#'     \text{out} = \text{input} + \text{other}
#' }
#' If `input` is of type FloatTensor or DoubleTensor, `other` must be
#' a real number, otherwise it should be an integer.
#'
#' 
#'
#' @param input (Tensor) the input tensor.
#' @param value (Number) the number to be added to each element of :attr:`input`
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4)
#' a
#' torch_add(a, 20)
#' }
#'
#' @name torch_add
#'
#' @export
NULL


#' Add
#'
#' add(input, alpha=1, other, out=None)
#' 
#' Each element of the tensor `other` is multiplied by the scalar
#' `alpha` and added to each element of the tensor `input`.
#' The resulting tensor is returned.
#' 
#' The shapes of `input` and `other` must be
#' :ref:`broadcastable <broadcasting-semantics>`.
#' 
#' \deqn{
#'     \text{out} = \text{input} + \text{alpha} \times \text{other}
#' }
#' If `other` is of type FloatTensor or DoubleTensor, `alpha` must be
#' a real number, otherwise it should be an integer.
#'
#' 
#'
#' @param input (Tensor) the first input tensor
#' @param alpha (Number) the scalar multiplier for :attr:`other`
#' @param other (Tensor) the second input tensor
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4)
#' a
#' b = torch_randn(4, 1)
#' b
#' torch_add(a, 10, b)
#' }
#'
#' @name torch_add
#'
#' @export
NULL


#' Addmv
#'
#' Performs a matrix-vector product of the matrix `mat` and
#' the vector `vec`.
#' The vector `input` is added to the final result.
#' 
#' If `mat` is a \eqn{(n \times m)} tensor, `vec` is a 1-D tensor of
#' size `m`, then `input` must be
#' :ref:`broadcastable <broadcasting-semantics>` with a 1-D tensor of size `n` and
#' `out` will be 1-D tensor of size `n`.
#' 
#' `alpha` and `beta` are scaling factors on matrix-vector product between
#' `mat` and `vec` and the added tensor `input` respectively.
#' 
#' \deqn{
#'     \text{out} = \beta\ \text{input} + \alpha\ (\text{mat} \mathbin{@} \text{vec})
#' }
#' For inputs of type `FloatTensor` or `DoubleTensor`, arguments `beta` and
#' `alpha` must be real numbers, otherwise they should be integers
#'
#' @section Signatures:
#' 
#' addmv(beta=1, input, alpha=1, mat, vec, out=None) -> Tensor
#'
#'
#' @param beta (Number, optional) multiplier for :attr:`input` (\eqn{\beta})
#' @param input (Tensor) vector to be added
#' @param alpha (Number, optional) multiplier for \eqn{mat @ vec} (\eqn{\alpha})
#' @param mat (Tensor) matrix to be multiplied
#' @param vec (Tensor) vector to be multiplied
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' M = torch_randn(2)
#' mat = torch_randn(2, 3)
#' vec = torch_randn(3)
#' torch_addmv(M, mat, vec)
#' }
#'
#' @name torch_addmv
#'
#' @export
NULL


#' Addr
#'
#' Performs the outer-product of vectors `vec1` and `vec2`
#' and adds it to the matrix `input`.
#' 
#' Optional values `beta` and `alpha` are scaling factors on the
#' outer product between `vec1` and `vec2` and the added matrix
#' `input` respectively.
#' 
#' \deqn{
#'     \text{out} = \beta\ \text{input} + \alpha\ (\text{vec1} \otimes \text{vec2})
#' }
#' If `vec1` is a vector of size `n` and `vec2` is a vector
#' of size `m`, then `input` must be
#' :ref:`broadcastable <broadcasting-semantics>` with a matrix of size
#' \eqn{(n \times m)} and `out` will be a matrix of size
#' \eqn{(n \times m)}.
#' 
#' For inputs of type `FloatTensor` or `DoubleTensor`, arguments `beta` and
#' `alpha` must be real numbers, otherwise they should be integers
#'
#' @section Signatures:
#' 
#' addr(beta=1, input, alpha=1, vec1, vec2, out=None) -> Tensor
#'
#'
#' @param beta (Number, optional) multiplier for :attr:`input` (\eqn{\beta})
#' @param input (Tensor) matrix to be added
#' @param alpha (Number, optional) multiplier for \eqn{\text{vec1} \otimes \text{vec2}} (\eqn{\alpha})
#' @param vec1 (Tensor) the first vector of the outer product
#' @param vec2 (Tensor) the second vector of the outer product
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' vec1 = torch_arange(1., 4.)
#' vec2 = torch_arange(1., 3.)
#' M = torch_zeros(3, 2)
#' torch_addr(M, vec1, vec2)
#' }
#'
#' @name torch_addr
#'
#' @export
NULL


#' Allclose
#'
#' This function checks if all `input` and `other` satisfy the condition:
#' 
#' \deqn{
#'     \lvert \text{input} - \text{other} \rvert \leq \texttt{atol} + \texttt{rtol} \times \lvert \text{other} \rvert
#' }
#' elementwise, for all elements of `input` and `other`. The behaviour of this function is analogous to
#' `numpy.allclose <https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html>`_
#'
#' @section Signatures:
#' 
#' allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False) -> bool
#'
#'
#' @param input (Tensor) first tensor to compare
#' @param other (Tensor) second tensor to compare
#' @param atol (float, optional) absolute tolerance. Default: 1e-08
#' @param rtol (float, optional) relative tolerance. Default: 1e-05
#' @param equal_nan (bool, optional) if ``True``, then two ``NaN`` s will be compared as equal. Default: ``False``
#'
#' @examples
#' \dontrun{
#' torch_allclose(torch_tensor([10000., 1e-07]), torch_tensor([10000.1, 1e-08]))
#' torch_allclose(torch_tensor([10000., 1e-08]), torch_tensor([10000.1, 1e-09]))
#' torch_allclose(torch_tensor([1.0, float('nan')]), torch_tensor([1.0, float('nan')]))
#' torch_allclose(torch_tensor([1.0, float('nan')]), torch_tensor([1.0, float('nan')]), equal_nan=True)
#' }
#'
#' @name torch_allclose
#'
#' @export
NULL


#' Arange
#'
#' Returns a 1-D tensor of size \eqn{\left\lceil \frac{\text{end} - \text{start}}{\text{step}} \right\rceil}
#' with values from the interval ``[start, end)`` taken with common difference
#' `step` beginning from `start`.
#' 
#' Note that non-integer `step` is subject to floating point rounding errors when
#' comparing against `end`; to avoid inconsistency, we advise adding a small epsilon to `end`
#' in such cases.
#' 
#' \deqn{
#'     \text{out}_{{i+1}} = \text{out}_{i} + \text{step}
#' }
#'
#' @section Signatures:
#' 
#' arange(start=0, end, step=1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
#'
#'
#' @param start (Number) the starting value for the set of points. Default: ``0``.
#' @param end (Number) the ending value for the set of points
#' @param step (Number) the gap between each pair of adjacent points. Default: ``1``.
#' @param out (Tensor, optional) the output tensor.
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of returned tensor.        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`). If `dtype` is not given, infer the data type from the other input        arguments. If any of `start`, `end`, or `stop` are floating-point, the        `dtype` is inferred to be the default dtype, see        :meth:`~torch.get_default_dtype`. Otherwise, the `dtype` is inferred to        be `torch.int64`.
#' @param layout (:class:`torch.layout`, optional) the desired layout of returned Tensor.        Default: ``torch.strided``.
#' @param device (:class:`torch.device`, optional) the desired device of returned tensor.        Default: if ``None``, uses the current device for the default tensor type        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU        for CPU tensor types and the current CUDA device for CUDA tensor types.
#' @param requires_grad (bool, optional) If autograd should record operations on the        returned tensor. Default: ``False``.
#'
#' @examples
#' \dontrun{
#' torch_arange(5)
#' torch_arange(1, 4)
#' torch_arange(1, 2.5, 0.5)
#' }
#'
#' @name torch_arange
#'
#' @export
NULL


#' Argmax
#'
#' Returns the indices of the maximum value of all elements in the `input` tensor.
#' 
#' This is the second value returned by :meth:`torch_max`. See its
#' documentation for the exact semantics of this method.
#'
#' @section Signatures:
#' 
#' argmax(input) -> LongTensor
#'
#'
#' @param input (Tensor) the input tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4, 4)
#' a
#' torch_argmax(a)
#' }
#'
#' @name torch_argmax
#'
#' @export
NULL


#' Argmax
#'
#' Returns the indices of the maximum values of a tensor across a dimension.
#' 
#' This is the second value returned by :meth:`torch_max`. See its
#' documentation for the exact semantics of this method.
#'
#' @section Signatures:
#' 
#' argmax(input, dim, keepdim=False) -> LongTensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param dim (int) the dimension to reduce. If ``None``, the argmax of the flattened input is returned.
#' @param keepdim (bool) whether the output tensor has :attr:`dim` retained or not. Ignored if ``dim=None``.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4, 4)
#' a
#' torch_argmax(a, dim=1)
#' }
#'
#' @name torch_argmax
#'
#' @export
NULL


#' Argmin
#'
#' Returns the indices of the minimum value of all elements in the `input` tensor.
#' 
#' This is the second value returned by :meth:`torch_min`. See its
#' documentation for the exact semantics of this method.
#'
#' @section Signatures:
#' 
#' argmin(input) -> LongTensor
#'
#'
#' @param input (Tensor) the input tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4, 4)
#' a
#' torch_argmin(a)
#' }
#'
#' @name torch_argmin
#'
#' @export
NULL


#' Argmin
#'
#' Returns the indices of the minimum values of a tensor across a dimension.
#' 
#' This is the second value returned by :meth:`torch_min`. See its
#' documentation for the exact semantics of this method.
#'
#' @section Signatures:
#' 
#' argmin(input, dim, keepdim=False, out=None) -> LongTensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param dim (int) the dimension to reduce. If ``None``, the argmin of the flattened input is returned.
#' @param keepdim (bool) whether the output tensor has :attr:`dim` retained or not. Ignored if ``dim=None``.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4, 4)
#' a
#' torch_argmin(a, dim=1)
#' }
#'
#' @name torch_argmin
#'
#' @export
NULL


#' As_strided
#'
#' Create a view of an existing `torch_Tensor` `input` with specified
#' `size`, `stride` and `storage_offset`.
#' 
#' @section Warning:
#'     More than one element of a created tensor may refer to a single memory
#'     location. As a result, in-place operations (especially ones that are
#'     vectorized) may result in incorrect behavior. If you need to write to
#'     the tensors, please clone them first.
#' 
#'     Many PyTorch functions, which return a view of a tensor, are internally
#'     implemented with this function. Those functions, like
#'     :meth:`torch_Tensor.expand`, are easier to read and are therefore more
#'     advisable to use.
#'
#' @section Signatures:
#' 
#' as_strided(input, size, stride, storage_offset=0) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param size (tuple or ints) the shape of the output tensor
#' @param stride (tuple or ints) the stride of the output tensor
#' @param storage_offset (int, optional) the offset in the underlying storage of the output tensor
#'
#' @examples
#' \dontrun{
#' x = torch_randn(3, 3)
#' x
#' t = torch_as_strided(x, (2, 2), (1, 2))
#' t
#' t = torch_as_strided(x, (2, 2), (1, 2), 1)
#' }
#'
#' @name torch_as_strided
#'
#' @export
NULL


#' Asin
#'
#' Returns a new tensor with the arcsine  of the elements of `input`.
#' 
#' \deqn{
#'     \text{out}_{i} = \sin^{-1}(\text{input}_{i})
#' }
#'
#' @section Signatures:
#' 
#' asin(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4)
#' a
#' torch_asin(a)
#' }
#'
#' @name torch_asin
#'
#' @export
NULL


#' Atan
#'
#' Returns a new tensor with the arctangent  of the elements of `input`.
#' 
#' \deqn{
#'     \text{out}_{i} = \tan^{-1}(\text{input}_{i})
#' }
#'
#' @section Signatures:
#' 
#' atan(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4)
#' a
#' torch_atan(a)
#' }
#'
#' @name torch_atan
#'
#' @export
NULL


#' Baddbmm
#'
#' Performs a batch matrix-matrix product of matrices in `batch1`
#' and `batch2`.
#' `input` is added to the final result.
#' 
#' `batch1` and `batch2` must be 3-D tensors each containing the same
#' number of matrices.
#' 
#' If `batch1` is a \eqn{(b \times n \times m)} tensor, `batch2` is a
#' \eqn{(b \times m \times p)} tensor, then `input` must be
#' :ref:`broadcastable <broadcasting-semantics>` with a
#' \eqn{(b \times n \times p)} tensor and `out` will be a
#' \eqn{(b \times n \times p)} tensor. Both `alpha` and `beta` mean the
#' same as the scaling factors used in :meth:`torch_addbmm`.
#' 
#' \deqn{
#'     \text{out}_i = \beta\ \text{input}_i + \alpha\ (\text{batch1}_i \mathbin{@} \text{batch2}_i)
#' }
#' For inputs of type `FloatTensor` or `DoubleTensor`, arguments `beta` and
#' `alpha` must be real numbers, otherwise they should be integers.
#'
#' @section Signatures:
#' 
#' baddbmm(beta=1, input, alpha=1, batch1, batch2, out=None) -> Tensor
#'
#'
#' @param beta (Number, optional) multiplier for :attr:`input` (\eqn{\beta})
#' @param input (Tensor) the tensor to be added
#' @param alpha (Number, optional) multiplier for \eqn{\text{batch1} \mathbin{@} \text{batch2}} (\eqn{\alpha})
#' @param batch1 (Tensor) the first batch of matrices to be multiplied
#' @param batch2 (Tensor) the second batch of matrices to be multiplied
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' M = torch_randn(10, 3, 5)
#' batch1 = torch_randn(10, 3, 4)
#' batch2 = torch_randn(10, 4, 5)
#' torch_baddbmm(M, batch1, batch2).size()
#' }
#'
#' @name torch_baddbmm
#'
#' @export
NULL


#' Bartlett_window
#'
#' Bartlett window function.
#' 
#' \deqn{
#'     w[n] = 1 - \left| \frac{2n}{N-1} - 1 \right| = \begin{cases}
#'         \frac{2n}{N - 1} & \text{if } 0 \leq n \leq \frac{N - 1}{2} \\
#'         2 - \frac{2n}{N - 1} & \text{if } \frac{N - 1}{2} < n < N \\
#'     \end{cases},
#' }
#' where \eqn{N} is the full window size.
#' 
#' The input `window_length` is a positive integer controlling the
#' returned window size. `periodic` flag determines whether the returned
#' window trims off the last duplicate value from the symmetric window and is
#' ready to be used as a periodic window with functions like
#' :meth:`torch_stft`. Therefore, if `periodic` is true, the \eqn{N} in
#' above formula is in fact \eqn{\text{window\_length} + 1}. Also, we always have
#' ``torch_bartlett_window(L, periodic=True)`` equal to
#' ``torch_bartlett_window(L + 1, periodic=False)[:-1])``.
#' 
#' @note
#'     If `window_length` \eqn{=1}, the returned window contains a single value 1.
#'
#' @section Signatures:
#' 
#' bartlett_window(window_length, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
#'
#'
#' @param window_length (int) the size of returned window
#' @param periodic (bool, optional) If True, returns a window to be used as periodic        function. If False, return a symmetric window.
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of returned tensor.        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`). Only floating point types are supported.
#' @param layout (:class:`torch.layout`, optional) the desired layout of returned window tensor. Only          ``torch.strided`` (dense layout) is supported.
#' @param device (:class:`torch.device`, optional) the desired device of returned tensor.        Default: if ``None``, uses the current device for the default tensor type        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU        for CPU tensor types and the current CUDA device for CUDA tensor types.
#' @param requires_grad (bool, optional) If autograd should record operations on the        returned tensor. Default: ``False``.
#'
#' 
#'
#' @name torch_bartlett_window
#'
#' @export
NULL


#' Bernoulli
#'
#' Draws binary random numbers (0 or 1) from a Bernoulli distribution.
#' 
#' The `input` tensor should be a tensor containing probabilities
#' to be used for drawing the binary random number.
#' Hence, all values in `input` have to be in the range:
#' \eqn{0 \leq \text{input}_i \leq 1}.
#' 
#' The \eqn{\text{i}^{th}} element of the output tensor will draw a
#' value \eqn{1} according to the \eqn{\text{i}^{th}} probability value given
#' in `input`.
#' 
#' \deqn{
#'     \text{out}_{i} \sim \mathrm{Bernoulli}(p = \text{input}_{i})
#' }
#' The returned `out` tensor only has values 0 or 1 and is of the same
#' shape as `input`.
#' 
#' `out` can have integral ``dtype``, but `input` must have floating
#' point ``dtype``.
#'
#' @section Signatures:
#' 
#' bernoulli(input, *, generator=None, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor of probability values for the Bernoulli distribution
#' @param generator (:class:`torch.Generator`, optional) a pseudorandom number generator for sampling
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_empty(3, 3).uniform_(0, 1)  # generate a uniform random matrix with range [0, 1]
#' a
#' torch_bernoulli(a)
#' a = torch_ones(3, 3) # probability of drawing "1" is 1
#' torch_bernoulli(a)
#' a = torch_zeros(3, 3) # probability of drawing "1" is 0
#' torch_bernoulli(a)
#' }
#'
#' @name torch_bernoulli
#'
#' @export
NULL


#' Bincount
#'
#' Count the frequency of each value in an array of non-negative ints.
#' 
#' The number of bins (size 1) is one larger than the largest value in
#' `input` unless `input` is empty, in which case the result is a
#' tensor of size 0. If `minlength` is specified, the number of bins is at least
#' `minlength` and if `input` is empty, then the result is tensor of size
#' `minlength` filled with zeros. If ``n`` is the value at position ``i``,
#' ``out[n] += weights[i]`` if `weights` is specified else
#' ``out[n] += 1``.
#' 
#' .. include:: cuda_deterministic.rst
#'
#' @section Signatures:
#' 
#' bincount(input, weights=None, minlength=0) -> Tensor
#'
#'
#' @param input (Tensor) 1-d int tensor
#' @param weights (Tensor) optional, weight for each value in the input tensor.        Should be of same size as input tensor.
#' @param minlength (int) optional, minimum number of bins. Should be non-negative.
#'
#' @examples
#' \dontrun{
#' input = torch_randint(0, 8, (5,), dtype=torch_int64)
#' weights = torch_linspace(0, 1, steps=5)
#' input, weights
#' torch_bincount(input)
#' input.bincount(weights)
#' }
#'
#' @name torch_bincount
#'
#' @export
NULL


#' Bitwise_not
#'
#' Computes the bitwise NOT of the given input tensor. The input tensor must be of
#' integral or Boolean types. For bool tensors, it computes the logical NOT.
#'
#' @section Signatures:
#' 
#' bitwise_not(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' 
#'
#' @name torch_bitwise_not
#'
#' @export
NULL


#' Logical_not
#'
#' Computes the element-wise logical NOT of the given input tensor. If not specified, the output tensor will have the bool
#' dtype. If the input tensor is not a bool tensor, zeros are treated as ``False`` and non-zeros are treated as ``True``.
#'
#' @section Signatures:
#' 
#' logical_not(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' torch_logical_not(torch_tensor([True, False]))
#' torch_logical_not(torch_tensor([0, 1, -10], dtype=torch_int8))
#' torch_logical_not(torch_tensor([0., 1.5, -10.], dtype=torch_double))
#' torch_logical_not(torch_tensor([0., 1., -10.], dtype=torch_double), out=torch_empty(3, dtype=torch_int16))
#' }
#'
#' @name torch_logical_not
#'
#' @export
NULL


#' Logical_xor
#'
#' Computes the element-wise logical XOR of the given input tensors. Zeros are treated as ``False`` and nonzeros are
#' treated as ``True``.
#'
#' @section Signatures:
#' 
#' logical_xor(input, other, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param other (Tensor) the tensor to compute XOR with
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' torch_logical_xor(torch_tensor([True, False, True]), torch_tensor([True, False, False]))
#' a = torch_tensor([0, 1, 10, 0], dtype=torch_int8)
#' b = torch_tensor([4, 0, 1, 0], dtype=torch_int8)
#' torch_logical_xor(a, b)
#' torch_logical_xor(a.double(), b.double())
#' torch_logical_xor(a.double(), b)
#' torch_logical_xor(a, b, out=torch_empty(4, dtype=torch_bool))
#' }
#'
#' @name torch_logical_xor
#'
#' @export
NULL


#' Blackman_window
#'
#' Blackman window function.
#' 
#' \deqn{
#'     w[n] = 0.42 - 0.5 \cos \left( \frac{2 \pi n}{N - 1} \right) + 0.08 \cos \left( \frac{4 \pi n}{N - 1} \right)
#' }
#' where \eqn{N} is the full window size.
#' 
#' The input `window_length` is a positive integer controlling the
#' returned window size. `periodic` flag determines whether the returned
#' window trims off the last duplicate value from the symmetric window and is
#' ready to be used as a periodic window with functions like
#' :meth:`torch_stft`. Therefore, if `periodic` is true, the \eqn{N} in
#' above formula is in fact \eqn{\text{window\_length} + 1}. Also, we always have
#' ``torch_blackman_window(L, periodic=True)`` equal to
#' ``torch_blackman_window(L + 1, periodic=False)[:-1])``.
#' 
#' @note
#'     If `window_length` \eqn{=1}, the returned window contains a single value 1.
#'
#' @section Signatures:
#' 
#' blackman_window(window_length, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
#'
#'
#' @param window_length (int) the size of returned window
#' @param periodic (bool, optional) If True, returns a window to be used as periodic        function. If False, return a symmetric window.
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of returned tensor.        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`). Only floating point types are supported.
#' @param layout (:class:`torch.layout`, optional) the desired layout of returned window tensor. Only          ``torch.strided`` (dense layout) is supported.
#' @param device (:class:`torch.device`, optional) the desired device of returned tensor.        Default: if ``None``, uses the current device for the default tensor type        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU        for CPU tensor types and the current CUDA device for CUDA tensor types.
#' @param requires_grad (bool, optional) If autograd should record operations on the        returned tensor. Default: ``False``.
#'
#' 
#'
#' @name torch_blackman_window
#'
#' @export
NULL


#' Bmm
#'
#' Performs a batch matrix-matrix product of matrices stored in `input`
#' and `mat2`.
#' 
#' `input` and `mat2` must be 3-D tensors each containing
#' the same number of matrices.
#' 
#' If `input` is a \eqn{(b \times n \times m)} tensor, `mat2` is a
#' \eqn{(b \times m \times p)} tensor, `out` will be a
#' \eqn{(b \times n \times p)} tensor.
#' 
#' \deqn{
#'     \text{out}_i = \text{input}_i \mathbin{@} \text{mat2}_i
#' }
#' @note This function does not :ref:`broadcast <broadcasting-semantics>`.
#'           For broadcasting matrix products, see [`torch_matmul`].
#'
#' @section Signatures:
#' 
#' bmm(input, mat2, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the first batch of matrices to be multiplied
#' @param mat2 (Tensor) the second batch of matrices to be multiplied
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' input = torch_randn(10, 3, 4)
#' mat2 = torch_randn(10, 4, 5)
#' res = torch_bmm(input, mat2)
#' res.size()
#' }
#'
#' @name torch_bmm
#'
#' @export
NULL


#' Broadcast_tensors
#'
#' Broadcasts the given tensors according to :ref:`broadcasting-semantics`.
#'
#' @section Signatures:
#' 
#' broadcast_tensors(*tensors) -> List of Tensors
#'
#'
#' @param *tensors NA any number of tensors of the same type
#'
#' 
#'
#' @name torch_broadcast_tensors
#'
#' @export
NULL


#' Cat
#'
#' Concatenates the given sequence of `seq` tensors in the given dimension.
#' All tensors must either have the same shape (except in the concatenating
#' dimension) or be empty.
#' 
#' [`torch_cat`] can be seen as an inverse operation for [`torch.split`]
#' and [`torch_chunk`].
#' 
#' [`torch_cat`] can be best understood via examples.
#'
#' @section Signatures:
#' 
#' cat(tensors, dim=0, out=None) -> Tensor
#'
#'
#' @param tensors (sequence of Tensors) any python sequence of tensors of the same type.        Non-empty tensors provided must have the same shape, except in the        cat dimension.
#' @param dim (int, optional) the dimension over which the tensors are concatenated
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' x = torch_randn(2, 3)
#' x
#' torch_cat((x, x, x), 0)
#' torch_cat((x, x, x), 1)
#' }
#'
#' @name torch_cat
#'
#' @export
NULL


#' Ceil
#'
#' Returns a new tensor with the ceil of the elements of `input`,
#' the smallest integer greater than or equal to each element.
#' 
#' \deqn{
#'     \text{out}_{i} = \left\lceil \text{input}_{i} \right\rceil = \left\lfloor \text{input}_{i} \right\rfloor + 1
#' }
#'
#' @section Signatures:
#' 
#' ceil(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4)
#' a
#' torch_ceil(a)
#' }
#'
#' @name torch_ceil
#'
#' @export
NULL


#' Chain_matmul
#'
#' Returns the matrix product of the \eqn{N} 2-D tensors. This product is efficiently computed
#'     using the matrix chain order algorithm which selects the order in which incurs the lowest cost in terms
#'     of arithmetic operations (`[CLRS]`_). Note that since this is a function to compute the product, \eqn{N}
#'     needs to be greater than or equal to 2; if equal to 2 then a trivial matrix-matrix product is returned.
#'     If \eqn{N} is 1, then this is a no-op - the original matrix is returned as is.
#'
#' 
#'
#' @param matrices (Tensors...) a sequence of 2 or more 2-D tensors whose product is to be determined.
#'
#' 
#'
#' @name torch_chain_matmul
#'
#' @export
NULL


#' Chunk
#'
#' Splits a tensor into a specific number of chunks.
#' 
#' Last chunk will be smaller if the tensor size along the given dimension
#' `dim` is not divisible by `chunks`.
#'
#' @section Signatures:
#' 
#' chunk(input, chunks, dim=0) -> List of Tensors
#'
#'
#' @param input (Tensor) the tensor to split
#' @param chunks (int) number of chunks to return
#' @param dim (int) dimension along which to split the tensor
#'
#' 
#'
#' @name torch_chunk
#'
#' @export
NULL


#' Clamp
#'
#' Clamp all elements in `input` into the range `[` `min`, `max` `]` and return
#' a resulting tensor:
#' 
#' \deqn{
#'     y_i = \begin{cases}
#'         \text{min} & \text{if } x_i < \text{min} \\
#'         x_i & \text{if } \text{min} \leq x_i \leq \text{max} \\
#'         \text{max} & \text{if } x_i > \text{max}
#'     \end{cases}
#' }
#' If `input` is of type `FloatTensor` or `DoubleTensor`, args `min`
#' and `max` must be real numbers, otherwise they should be integers.
#'
#' @section Signatures:
#' 
#' clamp(input, min, max, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param min (Number) lower-bound of the range to be clamped to
#' @param max (Number) upper-bound of the range to be clamped to
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4)
#' a
#' torch_clamp(a, min=-0.5, max=0.5)
#' }
#'
#' @name torch_clamp
#'
#' @export
NULL


#' Clamp
#'
#' Clamps all elements in `input` to be larger or equal `min`.
#' 
#' If `input` is of type `FloatTensor` or `DoubleTensor`, `value`
#' should be a real number, otherwise it should be an integer.
#'
#' @section Signatures:
#' 
#' clamp(input, *, min, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param value (Number) minimal value of each element in the output
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4)
#' a
#' torch_clamp(a, min=0.5)
#' }
#'
#' @name torch_clamp
#'
#' @export
NULL


#' Clamp
#'
#' Clamps all elements in `input` to be smaller or equal `max`.
#' 
#' If `input` is of type `FloatTensor` or `DoubleTensor`, `value`
#' should be a real number, otherwise it should be an integer.
#'
#' @section Signatures:
#' 
#' clamp(input, *, max, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param value (Number) maximal value of each element in the output
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4)
#' a
#' torch_clamp(a, max=0.5)
#' }
#'
#' @name torch_clamp
#'
#' @export
NULL


#' Conv1d
#'
#' Applies a 1D convolution over an input signal composed of several input
#' planes.
#' 
#' See :class:`~torch.nn.Conv1d` for details and output shape.
#' 
#' .. include:: cudnn_deterministic.rst
#'
#' @section Signatures:
#' 
#' conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor
#'
#'
#' @param input NA input tensor of shape \eqn{(\text{minibatch} , \text{in\_channels} , iW)}
#' @param weight NA filters of shape \eqn{(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kW)}
#' @param bias NA optional bias of shape \eqn{(\text{out\_channels})}. Default: ``None``
#' @param stride NA the stride of the convolving kernel. Can be a single number or      a one-element tuple `(sW,)`. Default: 1
#' @param padding NA implicit paddings on both sides of the input. Can be a      single number or a one-element tuple `(padW,)`. Default: 0
#' @param dilation NA the spacing between kernel elements. Can be a single number or      a one-element tuple `(dW,)`. Default: 1
#' @param groups NA split input into groups, \eqn{\text{in\_channels}} should be divisible by      the number of groups. Default: 1
#'
#' 
#'
#' @name torch_conv1d
#'
#' @export
NULL


#' Conv2d
#'
#' Applies a 2D convolution over an input image composed of several input
#' planes.
#' 
#' See :class:`~torch.nn.Conv2d` for details and output shape.
#' 
#' .. include:: cudnn_deterministic.rst
#'
#' @section Signatures:
#' 
#' conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor
#'
#'
#' @param input NA input tensor of shape \eqn{(\text{minibatch} , \text{in\_channels} , iH , iW)}
#' @param weight NA filters of shape \eqn{(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kH , kW)}
#' @param bias NA optional bias tensor of shape \eqn{(\text{out\_channels})}. Default: ``None``
#' @param stride NA the stride of the convolving kernel. Can be a single number or a      tuple `(sH, sW)`. Default: 1
#' @param padding NA implicit paddings on both sides of the input. Can be a      single number or a tuple `(padH, padW)`. Default: 0
#' @param dilation NA the spacing between kernel elements. Can be a single number or      a tuple `(dH, dW)`. Default: 1
#' @param groups NA split input into groups, \eqn{\text{in\_channels}} should be divisible by the      number of groups. Default: 1
#'
#' 
#'
#' @name torch_conv2d
#'
#' @export
NULL


#' Conv3d
#'
#' Applies a 3D convolution over an input image composed of several input
#' planes.
#' 
#' See :class:`~torch.nn.Conv3d` for details and output shape.
#' 
#' .. include:: cudnn_deterministic.rst
#'
#' @section Signatures:
#' 
#' conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor
#'
#'
#' @param input NA input tensor of shape \eqn{(\text{minibatch} , \text{in\_channels} , iT , iH , iW)}
#' @param weight NA filters of shape \eqn{(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kT , kH , kW)}
#' @param bias NA optional bias tensor of shape \eqn{(\text{out\_channels})}. Default: None
#' @param stride NA the stride of the convolving kernel. Can be a single number or a      tuple `(sT, sH, sW)`. Default: 1
#' @param padding NA implicit paddings on both sides of the input. Can be a      single number or a tuple `(padT, padH, padW)`. Default: 0
#' @param dilation NA the spacing between kernel elements. Can be a single number or      a tuple `(dT, dH, dW)`. Default: 1
#' @param groups NA split input into groups, \eqn{\text{in\_channels}} should be divisible by      the number of groups. Default: 1
#'
#' 
#'
#' @name torch_conv3d
#'
#' @export
NULL


#' Conv_tbc
#'
#' Applies a 1-dimensional sequence convolution over an input sequence.
#' Input and output dimensions are (Time, Batch, Channels) - hence TBC.
#'
#' 
#'
#' @param input NA input tensor of shape \eqn{(\text{sequence length} \times batch \times \text{in\_channels})}
#' @param weight NA filter of shape (\eqn{\text{kernel width} \times \text{in\_channels} \times \text{out\_channels}})
#' @param bias NA bias of shape (\eqn{\text{out\_channels}})
#' @param pad NA number of timesteps to pad. Default: 0
#'
#' 
#'
#' @name torch_conv_tbc
#'
#' @export
NULL


#' Conv_transpose1d
#'
#' Applies a 1D transposed convolution operator over an input signal
#' composed of several input planes, sometimes also called "deconvolution".
#' 
#' See :class:`~torch.nn.ConvTranspose1d` for details and output shape.
#' 
#' .. include:: cudnn_deterministic.rst
#'
#' @section Signatures:
#' 
#' conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor
#'
#'
#' @param input NA input tensor of shape \eqn{(\text{minibatch} , \text{in\_channels} , iW)}
#' @param weight NA filters of shape \eqn{(\text{in\_channels} , \frac{\text{out\_channels}}{\text{groups}} , kW)}
#' @param bias NA optional bias of shape \eqn{(\text{out\_channels})}. Default: None
#' @param stride NA the stride of the convolving kernel. Can be a single number or a      tuple ``(sW,)``. Default: 1
#' @param padding NA ``dilation * (kernel_size - 1) - padding`` zero-padding will be added to both      sides of each dimension in the input. Can be a single number or a tuple      ``(padW,)``. Default: 0
#' @param output_padding NA additional size added to one side of each dimension in the      output shape. Can be a single number or a tuple ``(out_padW)``. Default: 0
#' @param groups NA split input into groups, \eqn{\text{in\_channels}} should be divisible by the      number of groups. Default: 1
#' @param dilation NA the spacing between kernel elements. Can be a single number or      a tuple ``(dW,)``. Default: 1
#'
#' 
#'
#' @name torch_conv_transpose1d
#'
#' @export
NULL


#' Conv_transpose2d
#'
#' Applies a 2D transposed convolution operator over an input image
#' composed of several input planes, sometimes also called "deconvolution".
#' 
#' See :class:`~torch.nn.ConvTranspose2d` for details and output shape.
#' 
#' .. include:: cudnn_deterministic.rst
#'
#' @section Signatures:
#' 
#' conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor
#'
#'
#' @param input NA input tensor of shape \eqn{(\text{minibatch} , \text{in\_channels} , iH , iW)}
#' @param weight NA filters of shape \eqn{(\text{in\_channels} , \frac{\text{out\_channels}}{\text{groups}} , kH , kW)}
#' @param bias NA optional bias of shape \eqn{(\text{out\_channels})}. Default: None
#' @param stride NA the stride of the convolving kernel. Can be a single number or a      tuple ``(sH, sW)``. Default: 1
#' @param padding NA ``dilation * (kernel_size - 1) - padding`` zero-padding will be added to both      sides of each dimension in the input. Can be a single number or a tuple      ``(padH, padW)``. Default: 0
#' @param output_padding NA additional size added to one side of each dimension in the      output shape. Can be a single number or a tuple ``(out_padH, out_padW)``.      Default: 0
#' @param groups NA split input into groups, \eqn{\text{in\_channels}} should be divisible by the      number of groups. Default: 1
#' @param dilation NA the spacing between kernel elements. Can be a single number or      a tuple ``(dH, dW)``. Default: 1
#'
#' 
#'
#' @name torch_conv_transpose2d
#'
#' @export
NULL


#' Conv_transpose3d
#'
#' Applies a 3D transposed convolution operator over an input image
#' composed of several input planes, sometimes also called "deconvolution"
#' 
#' See :class:`~torch.nn.ConvTranspose3d` for details and output shape.
#' 
#' .. include:: cudnn_deterministic.rst
#'
#' @section Signatures:
#' 
#' conv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor
#'
#'
#' @param input NA input tensor of shape \eqn{(\text{minibatch} , \text{in\_channels} , iT , iH , iW)}
#' @param weight NA filters of shape \eqn{(\text{in\_channels} , \frac{\text{out\_channels}}{\text{groups}} , kT , kH , kW)}
#' @param bias NA optional bias of shape \eqn{(\text{out\_channels})}. Default: None
#' @param stride NA the stride of the convolving kernel. Can be a single number or a      tuple ``(sT, sH, sW)``. Default: 1
#' @param padding NA ``dilation * (kernel_size - 1) - padding`` zero-padding will be added to both      sides of each dimension in the input. Can be a single number or a tuple      ``(padT, padH, padW)``. Default: 0
#' @param output_padding NA additional size added to one side of each dimension in the      output shape. Can be a single number or a tuple      ``(out_padT, out_padH, out_padW)``. Default: 0
#' @param groups NA split input into groups, \eqn{\text{in\_channels}} should be divisible by the      number of groups. Default: 1
#' @param dilation NA the spacing between kernel elements. Can be a single number or      a tuple `(dT, dH, dW)`. Default: 1
#'
#' 
#'
#' @name torch_conv_transpose3d
#'
#' @export
NULL


#' Cos
#'
#' Returns a new tensor with the cosine  of the elements of `input`.
#' 
#' \deqn{
#'     \text{out}_{i} = \cos(\text{input}_{i})
#' }
#'
#' @section Signatures:
#' 
#' cos(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4)
#' a
#' torch_cos(a)
#' }
#'
#' @name torch_cos
#'
#' @export
NULL


#' Cosh
#'
#' Returns a new tensor with the hyperbolic cosine  of the elements of
#' `input`.
#' 
#' \deqn{
#'     \text{out}_{i} = \cosh(\text{input}_{i})
#' }
#'
#' @section Signatures:
#' 
#' cosh(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4)
#' a
#' torch_cosh(a)
#' }
#'
#' @name torch_cosh
#'
#' @export
NULL


#' Cumsum
#'
#' Returns the cumulative sum of elements of `input` in the dimension
#' `dim`.
#' 
#' For example, if `input` is a vector of size N, the result will also be
#' a vector of size N, with elements.
#' 
#' \deqn{
#'     y_i = x_1 + x_2 + x_3 + \dots + x_i
#' }
#'
#' @section Signatures:
#' 
#' cumsum(input, dim, out=None, dtype=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param dim (int) the dimension to do the operation over
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of returned tensor.        If specified, the input tensor is casted to :attr:`dtype` before the operation        is performed. This is useful for preventing data type overflows. Default: None.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(10)
#' a
#' torch_cumsum(a, dim=0)
#' }
#'
#' @name torch_cumsum
#'
#' @export
NULL


#' Cumprod
#'
#' Returns the cumulative product of elements of `input` in the dimension
#' `dim`.
#' 
#' For example, if `input` is a vector of size N, the result will also be
#' a vector of size N, with elements.
#' 
#' \deqn{
#'     y_i = x_1 \times x_2\times x_3\times \dots \times x_i
#' }
#'
#' @section Signatures:
#' 
#' cumprod(input, dim, out=None, dtype=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param dim (int) the dimension to do the operation over
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of returned tensor.        If specified, the input tensor is casted to :attr:`dtype` before the operation        is performed. This is useful for preventing data type overflows. Default: None.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(10)
#' a
#' torch_cumprod(a, dim=0)
#' a[5] = 0.0
#' torch_cumprod(a, dim=0)
#' }
#'
#' @name torch_cumprod
#'
#' @export
NULL


#' Det
#'
#' Calculates determinant of a square matrix or batches of square matrices.
#' 
#' @note
#'     Backward through :meth:`det` internally uses SVD results when `input` is
#'     not invertible. In this case, double backward through :meth:`det` will be
#'     unstable in when `input` doesn't have distinct singular values. See
#'     :meth:`~torch.svd` for details.
#'
#' @section Signatures:
#' 
#' det(input) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor of size ``(*, n, n)`` where ``*`` is zero or more                batch dimensions.
#'
#' @examples
#' \dontrun{
#' A = torch_randn(3, 3)
#' torch_det(A)
#' A = torch_randn(3, 2, 2)
#' A
#' A.det()
#' }
#'
#' @name torch_det
#'
#' @export
NULL


#' Diag_embed
#'
#' Creates a tensor whose diagonals of certain 2D planes (specified by
#' `dim1` and `dim2`) are filled by `input`.
#' To facilitate creating batched diagonal matrices, the 2D planes formed by
#' the last two dimensions of the returned tensor are chosen by default.
#' 
#' The argument `offset` controls which diagonal to consider:
#' 
#' - If `offset` = 0, it is the main diagonal.
#' - If `offset` > 0, it is above the main diagonal.
#' - If `offset` < 0, it is below the main diagonal.
#' 
#' The size of the new matrix will be calculated to make the specified diagonal
#' of the size of the last input dimension.
#' Note that for `offset` other than \eqn{0}, the order of `dim1`
#' and `dim2` matters. Exchanging them is equivalent to changing the
#' sign of `offset`.
#' 
#' Applying :meth:`torch_diagonal` to the output of this function with
#' the same arguments yields a matrix identical to input. However,
#' :meth:`torch_diagonal` has different default dimensions, so those
#' need to be explicitly specified.
#'
#' @section Signatures:
#' 
#' diag_embed(input, offset=0, dim1=-2, dim2=-1) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor. Must be at least 1-dimensional.
#' @param offset (int, optional) which diagonal to consider. Default: 0        (main diagonal).
#' @param dim1 (int, optional) first dimension with respect to which to        take diagonal. Default: -2.
#' @param dim2 (int, optional) second dimension with respect to which to        take diagonal. Default: -1.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(2, 3)
#' torch_diag_embed(a)
#' torch_diag_embed(a, offset=1, dim1=0, dim2=2)
#' }
#'
#' @name torch_diag_embed
#'
#' @export
NULL


#' Diagflat
#'
#' - If `input` is a vector (1-D tensor), then returns a 2-D square tensor
#'   with the elements of `input` as the diagonal.
#' - If `input` is a tensor with more than one dimension, then returns a
#'   2-D tensor with diagonal elements equal to a flattened `input`.
#' 
#' The argument `offset` controls which diagonal to consider:
#' 
#' - If `offset` = 0, it is the main diagonal.
#' - If `offset` > 0, it is above the main diagonal.
#' - If `offset` < 0, it is below the main diagonal.
#'
#' @section Signatures:
#' 
#' diagflat(input, offset=0) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param offset (int, optional) the diagonal to consider. Default: 0 (main        diagonal).
#'
#' 
#'
#' @name torch_diagflat
#'
#' @export
NULL


#' Diagonal
#'
#' Returns a partial view of `input` with the its diagonal elements
#' with respect to `dim1` and `dim2` appended as a dimension
#' at the end of the shape.
#' 
#' The argument `offset` controls which diagonal to consider:
#' 
#' - If `offset` = 0, it is the main diagonal.
#' - If `offset` > 0, it is above the main diagonal.
#' - If `offset` < 0, it is below the main diagonal.
#' 
#' Applying :meth:`torch_diag_embed` to the output of this function with
#' the same arguments yields a diagonal matrix with the diagonal entries
#' of the input. However, :meth:`torch_diag_embed` has different default
#' dimensions, so those need to be explicitly specified.
#'
#' @section Signatures:
#' 
#' diagonal(input, offset=0, dim1=0, dim2=1) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor. Must be at least 2-dimensional.
#' @param offset (int, optional) which diagonal to consider. Default: 0        (main diagonal).
#' @param dim1 (int, optional) first dimension with respect to which to        take diagonal. Default: 0.
#' @param dim2 (int, optional) second dimension with respect to which to        take diagonal. Default: 1.
#'
#' 
#'
#' @name torch_diagonal
#'
#' @export
NULL


#' Div
#'
#' Divides each element of the input ``input`` with the scalar ``other`` and
#' returns a new resulting tensor.
#' 
#' \deqn{
#'     \text{out}_i = \frac{\text{input}_i}{\text{other}}
#' }
#' If the :class:`torch_dtype` of ``input`` and ``other`` differ, the
#' :class:`torch_dtype` of the result tensor is determined following rules
#' described in the type promotion :ref:`documentation <type-promotion-doc>`. If
#' ``out`` is specified, the result must be :ref:`castable <type-promotion-doc>`
#' to the :class:`torch_dtype` of the specified output tensor. Integral division
#' by zero leads to undefined behavior.
#'
#' @section Signatures:
#' 
#' div(input, other, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param other (Number) the number to be divided to each element of ``input``
#'
#' @examples
#' \dontrun{
#' a = torch_randn(5)
#' a
#' torch_div(a, 0.5)
#' }
#'
#' @name torch_div
#'
#' @export
NULL


#' Div
#'
#' Each element of the tensor ``input`` is divided by each element of the tensor
#' ``other``. The resulting tensor is returned.
#' 
#' \deqn{
#'     \text{out}_i = \frac{\text{input}_i}{\text{other}_i}
#' }
#' The shapes of ``input`` and ``other`` must be :ref:`broadcastable
#' <broadcasting-semantics>`. If the :class:`torch_dtype` of ``input`` and
#' ``other`` differ, the :class:`torch_dtype` of the result tensor is determined
#' following rules described in the type promotion :ref:`documentation
#' <type-promotion-doc>`. If ``out`` is specified, the result must be
#' :ref:`castable <type-promotion-doc>` to the :class:`torch_dtype` of the
#' specified output tensor. Integral division by zero leads to undefined behavior.
#'
#' @section Signatures:
#' 
#' div(input, other, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the numerator tensor
#' @param other (Tensor) the denominator tensor
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4, 4)
#' a
#' b = torch_randn(4)
#' b
#' torch_div(a, b)
#' }
#'
#' @name torch_div
#'
#' @export
NULL


#' Dot
#'
#' Computes the dot product (inner product) of two tensors.
#' 
#' @note This function does not :ref:`broadcast <broadcasting-semantics>`.
#'
#' @section Signatures:
#' 
#' dot(input, tensor) -> Tensor
#'
#'
#'
#'
#' @examples
#' \dontrun{
#' torch_dot(torch_tensor([2, 3]), torch_tensor([2, 1]))
#' }
#'
#' @name torch_dot
#'
#' @export
NULL


#' Einsum
#'
#' This function provides a way of computing multilinear expressions (i.e. sums of products) using the
#' Einstein summation convention.
#'
#' @section Signatures:
#' 
#' einsum(equation, *operands) -> Tensor
#'
#'
#' @param equation (string) The equation is given in terms of lower case letters (indices) to be associated           with each dimension of the operands and result. The left hand side lists the operands           dimensions, separated by commas. There should be one index letter per tensor dimension.           The right hand side follows after `->` and gives the indices for the output.           If the `->` and right hand side are omitted, it implicitly defined as the alphabetically           sorted list of all indices appearing exactly once in the left hand side.           The indices not apprearing in the output are summed over after multiplying the operands           entries.           If an index appears several times for the same operand, a diagonal is taken.           Ellipses `...` represent a fixed number of dimensions. If the right hand side is inferred,           the ellipsis dimensions are at the beginning of the output.
#' @param operands (Tensor) The operands to compute the Einstein sum of.
#'
#' 
#'
#' @name torch_einsum
#'
#' @export
NULL


#' Empty
#'
#' Returns a tensor filled with uninitialized data. The shape of the tensor is
#' defined by the variable argument `size`.
#'
#' @section Signatures:
#' 
#' empty(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False) -> Tensor
#'
#'
#' @param size (int...) a sequence of integers defining the shape of the output tensor.        Can be a variable number of arguments or a collection like a list or tuple.
#' @param out (Tensor, optional) the output tensor.
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of returned tensor.        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
#' @param layout (:class:`torch.layout`, optional) the desired layout of returned Tensor.        Default: ``torch.strided``.
#' @param device (:class:`torch.device`, optional) the desired device of returned tensor.        Default: if ``None``, uses the current device for the default tensor type        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU        for CPU tensor types and the current CUDA device for CUDA tensor types.
#' @param requires_grad (bool, optional) If autograd should record operations on the        returned tensor. Default: ``False``.
#' @param pin_memory (bool, optional) If set, returned tensor would be allocated in        the pinned memory. Works only for CPU tensors. Default: ``False``.
#'
#' @examples
#' \dontrun{
#' torch_empty(2, 3)
#' }
#'
#' @name torch_empty
#'
#' @export
NULL


#' Empty_like
#'
#' Returns an uninitialized tensor with the same size as `input`.
#' ``torch_empty_like(input)`` is equivalent to
#' ``torch_empty(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)``.
#'
#' @section Signatures:
#' 
#' empty_like(input, dtype=None, layout=None, device=None, requires_grad=False) -> Tensor
#'
#'
#' @param input (Tensor) the size of :attr:`input` will determine size of the output tensor.
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of returned Tensor.        Default: if ``None``, defaults to the dtype of :attr:`input`.
#' @param layout (:class:`torch.layout`, optional) the desired layout of returned tensor.        Default: if ``None``, defaults to the layout of :attr:`input`.
#' @param device (:class:`torch.device`, optional) the desired device of returned tensor.        Default: if ``None``, defaults to the device of :attr:`input`.
#' @param requires_grad (bool, optional) If autograd should record operations on the        returned tensor. Default: ``False``.
#'
#' @examples
#' \dontrun{
#' torch_empty((2,3), dtype=torch_int64)
#' }
#'
#' @name torch_empty_like
#'
#' @export
NULL


#' Empty_strided
#'
#' Returns a tensor filled with uninitialized data. The shape and strides of the tensor is
#' defined by the variable argument `size` and `stride` respectively.
#' ``torch_empty_strided(size, stride)`` is equivalent to
#' ``torch_empty(size).as_strided(size, stride)``.
#' 
#' @section Warning:
#'     More than one element of the created tensor may refer to a single memory
#'     location. As a result, in-place operations (especially ones that are
#'     vectorized) may result in incorrect behavior. If you need to write to
#'     the tensors, please clone them first.
#'
#' @section Signatures:
#' 
#' empty_strided(size, stride, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False) -> Tensor
#'
#'
#' @param size (tuple of ints) the shape of the output tensor
#' @param stride (tuple of ints) the strides of the output tensor
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of returned tensor.        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
#' @param layout (:class:`torch.layout`, optional) the desired layout of returned Tensor.        Default: ``torch.strided``.
#' @param device (:class:`torch.device`, optional) the desired device of returned tensor.        Default: if ``None``, uses the current device for the default tensor type        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU        for CPU tensor types and the current CUDA device for CUDA tensor types.
#' @param requires_grad (bool, optional) If autograd should record operations on the        returned tensor. Default: ``False``.
#' @param pin_memory (bool, optional) If set, returned tensor would be allocated in        the pinned memory. Works only for CPU tensors. Default: ``False``.
#'
#' @examples
#' \dontrun{
#' a = torch_empty_strided((2, 3), (1, 2))
#' a
#' a.stride()
#' a.size()
#' }
#'
#' @name torch_empty_strided
#'
#' @export
NULL


#' Erf
#'
#' Computes the error function of each element. The error function is defined as follows:
#' 
#' \deqn{
#'     \mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2} dt
#' }
#'
#' @section Signatures:
#' 
#' erf(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' torch_erf(torch_tensor([0, -1., 10.]))
#' }
#'
#' @name torch_erf
#'
#' @export
NULL


#' Erfc
#'
#' Computes the complementary error function of each element of `input`.
#' The complementary error function is defined as follows:
#' 
#' \deqn{
#'     \mathrm{erfc}(x) = 1 - \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2} dt
#' }
#'
#' @section Signatures:
#' 
#' erfc(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' torch_erfc(torch_tensor([0, -1., 10.]))
#' }
#'
#' @name torch_erfc
#'
#' @export
NULL


#' Exp
#'
#' Returns a new tensor with the exponential of the elements
#' of the input tensor `input`.
#' 
#' \deqn{
#'     y_{i} = e^{x_{i}}
#' }
#'
#' @section Signatures:
#' 
#' exp(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' torch_exp(torch_tensor([0, math.log(2.)]))
#' }
#'
#' @name torch_exp
#'
#' @export
NULL


#' Expm1
#'
#' Returns a new tensor with the exponential of the elements minus 1
#' of `input`.
#' 
#' \deqn{
#'     y_{i} = e^{x_{i}} - 1
#' }
#'
#' @section Signatures:
#' 
#' expm1(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' torch_expm1(torch_tensor([0, math.log(2.)]))
#' }
#'
#' @name torch_expm1
#'
#' @export
NULL


#' Eye
#'
#' Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.
#'
#' @section Signatures:
#' 
#' eye(n, m=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
#'
#'
#' @param n (int) the number of rows
#' @param m (int, optional) the number of columns with default being :attr:`n`
#' @param out (Tensor, optional) the output tensor.
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of returned tensor.        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
#' @param layout (:class:`torch.layout`, optional) the desired layout of returned Tensor.        Default: ``torch.strided``.
#' @param device (:class:`torch.device`, optional) the desired device of returned tensor.        Default: if ``None``, uses the current device for the default tensor type        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU        for CPU tensor types and the current CUDA device for CUDA tensor types.
#' @param requires_grad (bool, optional) If autograd should record operations on the        returned tensor. Default: ``False``.
#'
#' @examples
#' \dontrun{
#' torch_eye(3)
#' }
#'
#' @name torch_eye
#'
#' @export
NULL


#' Flatten
#'
#' Flattens a contiguous range of dims in a tensor.
#'
#' @section Signatures:
#' 
#' flatten(input, start_dim=0, end_dim=-1) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param start_dim (int) the first dim to flatten
#' @param end_dim (int) the last dim to flatten
#'
#' @examples
#' \dontrun{
#' t = torch_tensor([[[1, 2],
#' torch_flatten(t)
#' torch_flatten(t, start_dim=1)
#' }
#'
#' @name torch_flatten
#'
#' @export
NULL


#' Floor
#'
#' Returns a new tensor with the floor of the elements of `input`,
#' the largest integer less than or equal to each element.
#' 
#' \deqn{
#'     \text{out}_{i} = \left\lfloor \text{input}_{i} \right\rfloor
#' }
#'
#' @section Signatures:
#' 
#' floor(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4)
#' a
#' torch_floor(a)
#' }
#'
#' @name torch_floor
#'
#' @export
NULL


#' Frac
#'
#' Computes the fractional portion of each element in `input`.
#' 
#' \deqn{
#'     \text{out}_{i} = \text{input}_{i} - \left\lfloor |\text{input}_{i}| \right\rfloor * \operatorname{sgn}(\text{input}_{i})
#' }
#'
#' @section Signatures:
#' 
#' frac(input, out=None) -> Tensor
#'
#'
#'
#'
#' @examples
#' \dontrun{
#' torch_frac(torch_tensor([1, 2.5, -3.2]))
#' }
#'
#' @name torch_frac
#'
#' @export
NULL


#' Full
#'
#' Returns a tensor of size `size` filled with `fill_value`.
#'
#' @section Signatures:
#' 
#' full(size, fill_value, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
#'
#'
#' @param size (int...) a list, tuple, or :class:`torch.Size` of integers defining the        shape of the output tensor.
#' @param fill_value NA the number to fill the output tensor with.
#' @param out (Tensor, optional) the output tensor.
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of returned tensor.        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
#' @param layout (:class:`torch.layout`, optional) the desired layout of returned Tensor.        Default: ``torch.strided``.
#' @param device (:class:`torch.device`, optional) the desired device of returned tensor.        Default: if ``None``, uses the current device for the default tensor type        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU        for CPU tensor types and the current CUDA device for CUDA tensor types.
#' @param requires_grad (bool, optional) If autograd should record operations on the        returned tensor. Default: ``False``.
#'
#' @examples
#' \dontrun{
#' torch_full((2, 3), 3.141592)
#' }
#'
#' @name torch_full
#'
#' @export
NULL


#' Full_like
#'
#' Returns a tensor with the same size as `input` filled with `fill_value`.
#' ``torch_full_like(input, fill_value)`` is equivalent to
#' ``torch_full(input.size(), fill_value, dtype=input.dtype, layout=input.layout, device=input.device)``.
#'
#' @section Signatures:
#' 
#' full_like(input, fill_value, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
#'
#'
#' @param input (Tensor) the size of :attr:`input` will determine size of the output tensor.
#' @param fill_value NA the number to fill the output tensor with.
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of returned Tensor.        Default: if ``None``, defaults to the dtype of :attr:`input`.
#' @param layout (:class:`torch.layout`, optional) the desired layout of returned tensor.        Default: if ``None``, defaults to the layout of :attr:`input`.
#' @param device (:class:`torch.device`, optional) the desired device of returned tensor.        Default: if ``None``, defaults to the device of :attr:`input`.
#' @param requires_grad (bool, optional) If autograd should record operations on the        returned tensor. Default: ``False``.
#'
#' 
#'
#' @name torch_full_like
#'
#' @export
NULL


#' Hann_window
#'
#' Hann window function.
#' 
#' \deqn{
#'     w[n] = \frac{1}{2}\ \left[1 - \cos \left( \frac{2 \pi n}{N - 1} \right)\right] =
#'             \sin^2 \left( \frac{\pi n}{N - 1} \right),
#' }
#' where \eqn{N} is the full window size.
#' 
#' The input `window_length` is a positive integer controlling the
#' returned window size. `periodic` flag determines whether the returned
#' window trims off the last duplicate value from the symmetric window and is
#' ready to be used as a periodic window with functions like
#' :meth:`torch_stft`. Therefore, if `periodic` is true, the \eqn{N} in
#' above formula is in fact \eqn{\text{window\_length} + 1}. Also, we always have
#' ``torch_hann_window(L, periodic=True)`` equal to
#' ``torch_hann_window(L + 1, periodic=False)[:-1])``.
#' 
#' @note
#'     If `window_length` \eqn{=1}, the returned window contains a single value 1.
#'
#' @section Signatures:
#' 
#' hann_window(window_length, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
#'
#'
#' @param window_length (int) the size of returned window
#' @param periodic (bool, optional) If True, returns a window to be used as periodic        function. If False, return a symmetric window.
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of returned tensor.        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`). Only floating point types are supported.
#' @param layout (:class:`torch.layout`, optional) the desired layout of returned window tensor. Only          ``torch.strided`` (dense layout) is supported.
#' @param device (:class:`torch.device`, optional) the desired device of returned tensor.        Default: if ``None``, uses the current device for the default tensor type        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU        for CPU tensor types and the current CUDA device for CUDA tensor types.
#' @param requires_grad (bool, optional) If autograd should record operations on the        returned tensor. Default: ``False``.
#'
#' 
#'
#' @name torch_hann_window
#'
#' @export
NULL


#' Hamming_window
#'
#' Hamming window function.
#' 
#' \deqn{
#'     w[n] = \alpha - \beta\ \cos \left( \frac{2 \pi n}{N - 1} \right),
#' }
#' where \eqn{N} is the full window size.
#' 
#' The input `window_length` is a positive integer controlling the
#' returned window size. `periodic` flag determines whether the returned
#' window trims off the last duplicate value from the symmetric window and is
#' ready to be used as a periodic window with functions like
#' :meth:`torch_stft`. Therefore, if `periodic` is true, the \eqn{N} in
#' above formula is in fact \eqn{\text{window\_length} + 1}. Also, we always have
#' ``torch_hamming_window(L, periodic=True)`` equal to
#' ``torch_hamming_window(L + 1, periodic=False)[:-1])``.
#' 
#' @note
#'     If `window_length` \eqn{=1}, the returned window contains a single value 1.
#' 
#' @note
#'     This is a generalized version of :meth:`torch_hann_window`.
#'
#' @section Signatures:
#' 
#' hamming_window(window_length, periodic=True, alpha=0.54, beta=0.46, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
#'
#'
#' @param window_length (int) the size of returned window
#' @param periodic (bool, optional) If True, returns a window to be used as periodic        function. If False, return a symmetric window.
#' @param alpha (float, optional) The coefficient \eqn{\alpha} in the equation above
#' @param beta (float, optional) The coefficient \eqn{\beta} in the equation above
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of returned tensor.        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`). Only floating point types are supported.
#' @param layout (:class:`torch.layout`, optional) the desired layout of returned window tensor. Only          ``torch.strided`` (dense layout) is supported.
#' @param device (:class:`torch.device`, optional) the desired device of returned tensor.        Default: if ``None``, uses the current device for the default tensor type        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU        for CPU tensor types and the current CUDA device for CUDA tensor types.
#' @param requires_grad (bool, optional) If autograd should record operations on the        returned tensor. Default: ``False``.
#'
#' 
#'
#' @name torch_hamming_window
#'
#' @export
NULL


#' Ger
#'
#' Outer product of `input` and `vec2`.
#' If `input` is a vector of size \eqn{n} and `vec2` is a vector of
#' size \eqn{m}, then `out` must be a matrix of size \eqn{(n \times m)}.
#' 
#' @note This function does not :ref:`broadcast <broadcasting-semantics>`.
#'
#' @section Signatures:
#' 
#' ger(input, vec2, out=None) -> Tensor
#'
#'
#' @param input (Tensor) 1-D input vector
#' @param vec2 (Tensor) 1-D input vector
#' @param out (Tensor, optional) optional output matrix
#'
#' @examples
#' \dontrun{
#' v1 = torch_arange(1., 5.)
#' v2 = torch_arange(1., 4.)
#' torch_ger(v1, v2)
#' }
#'
#' @name torch_ger
#'
#' @export
NULL


#' Fft
#'
#' Complex-to-complex Discrete Fourier Transform
#' 
#' This method computes the complex-to-complex discrete Fourier transform.
#' Ignoring the batch dimensions, it computes the following expression:
#' 
#' \deqn{
#'     X[\omega_1, \dots, \omega_d] =
#'         \sum_{n_1=0}^{N_1-1} \dots \sum_{n_d=0}^{N_d-1} x[n_1, \dots, n_d]
#'          e^{-j\ 2 \pi \sum_{i=0}^d \frac{\omega_i n_i}{N_i}},
#' }
#' where \eqn{d} = `signal_ndim` is number of dimensions for the
#' signal, and \eqn{N_i} is the size of signal dimension \eqn{i}.
#' 
#' This method supports 1D, 2D and 3D complex-to-complex transforms, indicated
#' by `signal_ndim`. `input` must be a tensor with last dimension
#' of size 2, representing the real and imaginary components of complex
#' numbers, and should have at least ``signal_ndim + 1`` dimensions with optionally
#' arbitrary number of leading batch dimensions. If `normalized` is set to
#' ``True``, this normalizes the result by dividing it with
#' \eqn{\sqrt{\prod_{i=1}^K N_i}} so that the operator is unitary.
#' 
#' Returns the real and the imaginary parts together as one tensor of the same
#' shape of `input`.
#' 
#' The inverse of this function is [`~torch.ifft`].
#' 
#' @note
#'     For CUDA tensors, an LRU cache is used for cuFFT plans to speed up
#'     repeatedly running FFT methods on tensors of same geometry with same
#'     configuration. See :ref:`cufft-plan-cache` for more details on how to
#'     monitor and control the cache.
#' 
#' @section Warning:
#'     For CPU tensors, this method is currently only available with MKL. Use
#'     [`torch_backends.mkl.is_available`] to check if MKL is installed.
#'
#' @section Signatures:
#' 
#' fft(input, signal_ndim, normalized=False) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor of at least :attr:`signal_ndim` ``+ 1``        dimensions
#' @param signal_ndim (int) the number of dimensions in each signal.        :attr:`signal_ndim` can only be 1, 2 or 3
#' @param normalized (bool, optional) controls whether to return normalized results.        Default: ``False``
#'
#' @examples
#' \dontrun{
#' # unbatched 2D FFT
#' x = torch_randn(4, 3, 2)
#' torch_fft(x, 2)
#' # batched 1D FFT
#' torch_fft(x, 1)
#' # arbitrary number of batch dimensions, 2D FFT
#' x = torch_randn(3, 3, 5, 5, 2)
#' y = torch_fft(x, 2)
#' y.shape
#' }
#'
#' @name torch_fft
#'
#' @export
NULL


#' Ifft
#'
#' Complex-to-complex Inverse Discrete Fourier Transform
#' 
#' This method computes the complex-to-complex inverse discrete Fourier
#' transform. Ignoring the batch dimensions, it computes the following
#' expression:
#' 
#' \deqn{
#'     X[\omega_1, \dots, \omega_d] =
#'         \frac{1}{\prod_{i=1}^d N_i} \sum_{n_1=0}^{N_1-1} \dots \sum_{n_d=0}^{N_d-1} x[n_1, \dots, n_d]
#'          e^{\ j\ 2 \pi \sum_{i=0}^d \frac{\omega_i n_i}{N_i}},
#' }
#' where \eqn{d} = `signal_ndim` is number of dimensions for the
#' signal, and \eqn{N_i} is the size of signal dimension \eqn{i}.
#' 
#' The argument specifications are almost identical with [`~torch.fft`].
#' However, if `normalized` is set to ``True``, this instead returns the
#' results multiplied by \eqn{\sqrt{\prod_{i=1}^d N_i}}, to become a unitary
#' operator. Therefore, to invert a [`~torch.fft`], the `normalized`
#' argument should be set identically for [`~torch.fft`].
#' 
#' Returns the real and the imaginary parts together as one tensor of the same
#' shape of `input`.
#' 
#' The inverse of this function is [`~torch.fft`].
#' 
#' @note
#'     For CUDA tensors, an LRU cache is used for cuFFT plans to speed up
#'     repeatedly running FFT methods on tensors of same geometry with same
#'     configuration. See :ref:`cufft-plan-cache` for more details on how to
#'     monitor and control the cache.
#' 
#' @section Warning:
#'     For CPU tensors, this method is currently only available with MKL. Use
#'     [`torch_backends.mkl.is_available`] to check if MKL is installed.
#'
#' @section Signatures:
#' 
#' ifft(input, signal_ndim, normalized=False) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor of at least :attr:`signal_ndim` ``+ 1``        dimensions
#' @param signal_ndim (int) the number of dimensions in each signal.        :attr:`signal_ndim` can only be 1, 2 or 3
#' @param normalized (bool, optional) controls whether to return normalized results.        Default: ``False``
#'
#' @examples
#' \dontrun{
#' x = torch_randn(3, 3, 2)
#' x
#' y = torch_fft(x, 2)
#' torch_ifft(y, 2)  # recover x
#' }
#'
#' @name torch_ifft
#'
#' @export
NULL


#' Rfft
#'
#' Real-to-complex Discrete Fourier Transform
#' 
#' This method computes the real-to-complex discrete Fourier transform. It is
#' mathematically equivalent with [`~torch.fft`] with differences only in
#' formats of the input and output.
#' 
#' This method supports 1D, 2D and 3D real-to-complex transforms, indicated
#' by `signal_ndim`. `input` must be a tensor with at least
#' ``signal_ndim`` dimensions with optionally arbitrary number of leading batch
#' dimensions. If `normalized` is set to ``True``, this normalizes the result
#' by dividing it with \eqn{\sqrt{\prod_{i=1}^K N_i}} so that the operator is
#' unitary, where \eqn{N_i} is the size of signal dimension \eqn{i}.
#' 
#' The real-to-complex Fourier transform results follow conjugate symmetry:
#' 
#' \deqn{
#'     X[\omega_1, \dots, \omega_d] = X^*[N_1 - \omega_1, \dots, N_d - \omega_d],
#' }
#' where the index arithmetic is computed modulus the size of the corresponding
#' dimension, \eqn{\ ^*} is the conjugate operator, and
#' \eqn{d} = `signal_ndim`. `onesided` flag controls whether to avoid
#' redundancy in the output results. If set to ``True`` (default), the output will
#' not be full complex result of shape \eqn{(*, 2)}, where \eqn{*} is the shape
#' of `input`, but instead the last dimension will be halfed as of size
#' \eqn{\lfloor \frac{N_d}{2} \rfloor + 1}.
#' 
#' The inverse of this function is [`~torch.irfft`].
#' 
#' @note
#'     For CUDA tensors, an LRU cache is used for cuFFT plans to speed up
#'     repeatedly running FFT methods on tensors of same geometry with same
#'     configuration. See :ref:`cufft-plan-cache` for more details on how to
#'     monitor and control the cache.
#' 
#' @section Warning:
#'     For CPU tensors, this method is currently only available with MKL. Use
#'     [`torch_backends.mkl.is_available`] to check if MKL is installed.
#'
#' @section Signatures:
#' 
#' rfft(input, signal_ndim, normalized=False, onesided=True) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor of at least :attr:`signal_ndim` dimensions
#' @param signal_ndim (int) the number of dimensions in each signal.        :attr:`signal_ndim` can only be 1, 2 or 3
#' @param normalized (bool, optional) controls whether to return normalized results.        Default: ``False``
#' @param onesided (bool, optional) controls whether to return half of results to        avoid redundancy. Default: ``True``
#'
#' @examples
#' \dontrun{
#' x = torch_randn(5, 5)
#' torch_rfft(x, 2).shape
#' torch_rfft(x, 2, onesided=False).shape
#' }
#'
#' @name torch_rfft
#'
#' @export
NULL


#' Irfft
#'
#' Complex-to-real Inverse Discrete Fourier Transform
#' 
#' This method computes the complex-to-real inverse discrete Fourier transform.
#' It is mathematically equivalent with [`ifft`] with differences only in
#' formats of the input and output.
#' 
#' The argument specifications are almost identical with [`~torch.ifft`].
#' Similar to [`~torch.ifft`], if `normalized` is set to ``True``,
#' this normalizes the result by multiplying it with
#' \eqn{\sqrt{\prod_{i=1}^K N_i}} so that the operator is unitary, where
#' \eqn{N_i} is the size of signal dimension \eqn{i}.
#' 
#' @note
#'     Due to the conjugate symmetry, `input` do not need to contain the full
#'     complex frequency values. Roughly half of the values will be sufficient, as
#'     is the case when `input` is given by [`~torch.rfft`] with
#'     ``rfft(signal, onesided=True)``. In such case, set the `onesided`
#'     argument of this method to ``True``. Moreover, the original signal shape
#'     information can sometimes be lost, optionally set `signal_sizes` to be
#'     the size of the original signal (without the batch dimensions if in batched
#'     mode) to recover it with correct shape.
#' 
#'     Therefore, to invert an [`~torch.rfft`], the `normalized` and
#'     `onesided` arguments should be set identically for [`~torch.irfft`],
#'     and preferrably a `signal_sizes` is given to avoid size mismatch. See the
#'     example below for a case of size mismatch.
#' 
#'     See [`~torch.rfft`] for details on conjugate symmetry.
#' 
#' The inverse of this function is [`~torch.rfft`].
#' 
#' @section Warning:
#'     Generally speaking, input to this function should contain values
#'     following conjugate symmetry. Note that even if `onesided` is
#'     ``True``, often symmetry on some part is still needed. When this
#'     requirement is not satisfied, the behavior of [`~torch.irfft`] is
#'     undefined. Since [`torch_autograd.gradcheck`] estimates numerical
#'     Jacobian with point perturbations, [`~torch.irfft`] will almost
#'     certainly fail the check.
#' 
#' @note
#'     For CUDA tensors, an LRU cache is used for cuFFT plans to speed up
#'     repeatedly running FFT methods on tensors of same geometry with same
#'     configuration. See :ref:`cufft-plan-cache` for more details on how to
#'     monitor and control the cache.
#' 
#' @section Warning:
#'     For CPU tensors, this method is currently only available with MKL. Use
#'     [`torch_backends.mkl.is_available`] to check if MKL is installed.
#'
#' @section Signatures:
#' 
#' irfft(input, signal_ndim, normalized=False, onesided=True, signal_sizes=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor of at least :attr:`signal_ndim` ``+ 1``        dimensions
#' @param signal_ndim (int) the number of dimensions in each signal.        :attr:`signal_ndim` can only be 1, 2 or 3
#' @param normalized (bool, optional) controls whether to return normalized results.        Default: ``False``
#' @param onesided (bool, optional) controls whether :attr:`input` was halfed to avoid        redundancy, e.g., by :func:`rfft`. Default: ``True``
#' @param signal_sizes (list or :class:`torch.Size`, optional) the size of the original        signal (without batch dimension). Default: ``None``
#'
#' @examples
#' \dontrun{
#' x = torch_randn(4, 4)
#' torch_rfft(x, 2, onesided=True).shape
#'     >>>
#' # notice that with onesided=True, output size does not determine the original signal size
#' x = torch_randn(4, 5)
#' torch_rfft(x, 2, onesided=True).shape
#'     >>>
#' # now we use the original shape to recover x
#' x
#' y = torch_rfft(x, 2, onesided=True)
#' torch_irfft(y, 2, onesided=True, signal_sizes=x.shape)  # recover x
#' }
#'
#' @name torch_irfft
#'
#' @export
NULL


#' Inverse
#'
#' Takes the inverse of the square matrix `input`. `input` can be batches
#' of 2D square tensors, in which case this function would return a tensor composed of
#' individual inverses.
#' 
#' @note
#' 
#'     Irrespective of the original strides, the returned tensors will be
#'     transposed, i.e. with strides like `input.contiguous().transpose(-2, -1).stride()`
#'
#' @section Signatures:
#' 
#' inverse(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor of size \eqn{(*, n, n)} where `*` is zero or more                    batch dimensions
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' x = torch_rand(4, 4)
#' y = torch_inverse(x)
#' z = torch_mm(x, y)
#' z
#' torch_max(torch_abs(z - torch_eye(4))) # Max non-zero
#' # Batched inverse example
#' x = torch_randn(2, 3, 4, 4)
#' y = torch_inverse(x)
#' z = torch_matmul(x, y)
#' torch_max(torch_abs(z - torch_eye(4).expand_as(x))) # Max non-zero
#' }
#'
#' @name torch_inverse
#'
#' @export
NULL


#' Isnan
#'
#' Returns a new tensor with boolean elements representing if each element is `NaN` or not.
#'
#' 
#'
#' @param input (Tensor) A tensor to check
#'
#' @examples
#' \dontrun{
#' torch_isnan(torch_tensor([1, float('nan'), 2]))
#' }
#'
#' @name torch_isnan
#'
#' @export
NULL


#' Is_floating_point
#'
#' Returns True if the data type of `input` is a floating point data type i.e.,
#' one of ``torch_float64``, ``torch.float32`` and ``torch.float16``.
#'
#' @section Signatures:
#' 
#' is_floating_point(input) -> (bool)
#'
#'
#' @param input (Tensor) the PyTorch tensor to test
#'
#' 
#'
#' @name torch_is_floating_point
#'
#' @export
NULL


#' Kthvalue
#'
#' Returns a namedtuple ``(values, indices)`` where ``values`` is the `k` th
#' smallest element of each row of the `input` tensor in the given dimension
#' `dim`. And ``indices`` is the index location of each element found.
#' 
#' If `dim` is not given, the last dimension of the `input` is chosen.
#' 
#' If `keepdim` is ``True``, both the `values` and `indices` tensors
#' are the same size as `input`, except in the dimension `dim` where
#' they are of size 1. Otherwise, `dim` is squeezed
#' (see [`torch_squeeze`]), resulting in both the `values` and
#' `indices` tensors having 1 fewer dimension than the `input` tensor.
#'
#' @section Signatures:
#' 
#' kthvalue(input, k, dim=None, keepdim=False, out=None) -> (Tensor, LongTensor)
#'
#'
#' @param input (Tensor) the input tensor.
#' @param k (int) k for the k-th smallest element
#' @param dim (int, optional) the dimension to find the kth value along
#' @param keepdim (bool) whether the output tensor has :attr:`dim` retained or not.
#' @param out (tuple, optional) the output tuple of (Tensor, LongTensor)                           can be optionally given to be used as output buffers
#'
#' @examples
#' \dontrun{
#' x = torch_arange(1., 6.)
#' x
#' torch_kthvalue(x, 4)
#' x=torch_arange(1.,7.).resize_(2,3)
#' x
#' torch_kthvalue(x, 2, 0, True)
#' }
#'
#' @name torch_kthvalue
#'
#' @export
NULL


#' Linspace
#'
#' Returns a one-dimensional tensor of `steps`
#' equally spaced points between `start` and `end`.
#' 
#' The output tensor is 1-D of size `steps`.
#'
#' @section Signatures:
#' 
#' linspace(start, end, steps=100, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
#'
#'
#' @param start (float) the starting value for the set of points
#' @param end (float) the ending value for the set of points
#' @param steps (int) number of points to sample between :attr:`start`        and :attr:`end`. Default: ``100``.
#' @param out (Tensor, optional) the output tensor.
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of returned tensor.        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
#' @param layout (:class:`torch.layout`, optional) the desired layout of returned Tensor.        Default: ``torch.strided``.
#' @param device (:class:`torch.device`, optional) the desired device of returned tensor.        Default: if ``None``, uses the current device for the default tensor type        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU        for CPU tensor types and the current CUDA device for CUDA tensor types.
#' @param requires_grad (bool, optional) If autograd should record operations on the        returned tensor. Default: ``False``.
#'
#' @examples
#' \dontrun{
#' torch_linspace(3, 10, steps=5)
#' torch_linspace(-10, 10, steps=5)
#' torch_linspace(start=-10, end=10, steps=5)
#' torch_linspace(start=-10, end=10, steps=1)
#' }
#'
#' @name torch_linspace
#'
#' @export
NULL


#' Log
#'
#' Returns a new tensor with the natural logarithm of the elements
#' of `input`.
#' 
#' \deqn{
#'     y_{i} = \log_{e} (x_{i})
#' }
#'
#' @section Signatures:
#' 
#' log(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(5)
#' a
#' torch_log(a)
#' }
#'
#' @name torch_log
#'
#' @export
NULL


#' Log10
#'
#' Returns a new tensor with the logarithm to the base 10 of the elements
#' of `input`.
#' 
#' \deqn{
#'     y_{i} = \log_{10} (x_{i})
#' }
#'
#' @section Signatures:
#' 
#' log10(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_rand(5)
#' a
#' torch_log10(a)
#' }
#'
#' @name torch_log10
#'
#' @export
NULL


#' Log1p
#'
#' Returns a new tensor with the natural logarithm of (1 + `input`).
#' 
#' \deqn{
#'     y_i = \log_{e} (x_i + 1)
#' }
#' @note This function is more accurate than [`torch_log`] for small
#'           values of `input`
#'
#' @section Signatures:
#' 
#' log1p(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(5)
#' a
#' torch_log1p(a)
#' }
#'
#' @name torch_log1p
#'
#' @export
NULL


#' Log2
#'
#' Returns a new tensor with the logarithm to the base 2 of the elements
#' of `input`.
#' 
#' \deqn{
#'     y_{i} = \log_{2} (x_{i})
#' }
#'
#' @section Signatures:
#' 
#' log2(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_rand(5)
#' a
#' torch_log2(a)
#' }
#'
#' @name torch_log2
#'
#' @export
NULL


#' Logdet
#'
#' Calculates log determinant of a square matrix or batches of square matrices.
#' 
#' @note
#'     Result is ``-inf`` if `input` has zero log determinant, and is ``nan`` if
#'     `input` has negative determinant.
#' 
#' @note
#'     Backward through :meth:`logdet` internally uses SVD results when `input`
#'     is not invertible. In this case, double backward through :meth:`logdet` will
#'     be unstable in when `input` doesn't have distinct singular values. See
#'     :meth:`~torch.svd` for details.
#'
#' @section Signatures:
#' 
#' logdet(input) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor of size ``(*, n, n)`` where ``*`` is zero or more                batch dimensions.
#'
#' @examples
#' \dontrun{
#' A = torch_randn(3, 3)
#' torch_det(A)
#' torch_logdet(A)
#' A
#' A.det()
#' A.det().log()
#' }
#'
#' @name torch_logdet
#'
#' @export
NULL


#' Logspace
#'
#' Returns a one-dimensional tensor of `steps` points
#' logarithmically spaced with base `base` between
#' \eqn{{\text{base}}^{\text{start}}} and \eqn{{\text{base}}^{\text{end}}}.
#' 
#' The output tensor is 1-D of size `steps`.
#'
#' @section Signatures:
#' 
#' logspace(start, end, steps=100, base=10.0, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
#'
#'
#' @param start (float) the starting value for the set of points
#' @param end (float) the ending value for the set of points
#' @param steps (int) number of points to sample between :attr:`start`        and :attr:`end`. Default: ``100``.
#' @param base (float) base of the logarithm function. Default: ``10.0``.
#' @param out (Tensor, optional) the output tensor.
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of returned tensor.        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
#' @param layout (:class:`torch.layout`, optional) the desired layout of returned Tensor.        Default: ``torch.strided``.
#' @param device (:class:`torch.device`, optional) the desired device of returned tensor.        Default: if ``None``, uses the current device for the default tensor type        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU        for CPU tensor types and the current CUDA device for CUDA tensor types.
#' @param requires_grad (bool, optional) If autograd should record operations on the        returned tensor. Default: ``False``.
#'
#' @examples
#' \dontrun{
#' torch_logspace(start=-10, end=10, steps=5)
#' torch_logspace(start=0.1, end=1.0, steps=5)
#' torch_logspace(start=0.1, end=1.0, steps=1)
#' torch_logspace(start=2, end=2, steps=1, base=2)
#' }
#'
#' @name torch_logspace
#'
#' @export
NULL


#' Logsumexp
#'
#' logsumexp(input, dim, keepdim=False, out=None)
#' 
#' Returns the log of summed exponentials of each row of the `input`
#' tensor in the given dimension `dim`. The computation is numerically
#' stabilized.
#' 
#' For summation index \eqn{j} given by `dim` and other indices \eqn{i}, the result is
#' 
#' \deqn{
#'         \text{logsumexp}(x)_{i} = \log \sum_j \exp(x_{ij})
#' }
#' 
#' If `keepdim` is ``True``, the output tensor is of the same size
#' as `input` except in the dimension(s) `dim` where it is of size 1.
#' Otherwise, `dim` is squeezed (see [`torch_squeeze`]), resulting in the
#' output tensor having 1 (or ``len(dim)``) fewer dimension(s).
#'
#' 
#'
#' @param input (Tensor) the input tensor.
#' @param dim (int or tuple of ints) the dimension or dimensions to reduce.
#' @param keepdim (bool) whether the output tensor has :attr:`dim` retained or not.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(3, 3)
#' torch_logsumexp(a, 1)
#' }
#'
#' @name torch_logsumexp
#'
#' @export
NULL


#' Matmul
#'
#' Matrix product of two tensors.
#' 
#' The behavior depends on the dimensionality of the tensors as follows:
#' 
#' - If both tensors are 1-dimensional, the dot product (scalar) is returned.
#' - If both arguments are 2-dimensional, the matrix-matrix product is returned.
#' - If the first argument is 1-dimensional and the second argument is 2-dimensional,
#'   a 1 is prepended to its dimension for the purpose of the matrix multiply.
#'   After the matrix multiply, the prepended dimension is removed.
#' - If the first argument is 2-dimensional and the second argument is 1-dimensional,
#'   the matrix-vector product is returned.
#' - If both arguments are at least 1-dimensional and at least one argument is
#'   N-dimensional (where N > 2), then a batched matrix multiply is returned.  If the first
#'   argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the
#'   batched matrix multiply and removed after.  If the second argument is 1-dimensional, a
#'   1 is appended to its dimension for the purpose of the batched matrix multiple and removed after.
#'   The non-matrix (i.e. batch) dimensions are :ref:`broadcasted <broadcasting-semantics>` (and thus
#'   must be broadcastable).  For example, if `input` is a
#'   \eqn{(j \times 1 \times n \times m)} tensor and `other` is a \eqn{(k \times m \times p)}
#'   tensor, `out` will be an \eqn{(j \times k \times n \times p)} tensor.
#' 
#' @note
#' 
#'     The 1-dimensional dot product version of this function does not support an `out` parameter.
#'
#' @section Signatures:
#' 
#' matmul(input, other, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the first tensor to be multiplied
#' @param other (Tensor) the second tensor to be multiplied
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' # vector x vector
#' tensor1 = torch_randn(3)
#' tensor2 = torch_randn(3)
#' torch_matmul(tensor1, tensor2).size()
#' # matrix x vector
#' tensor1 = torch_randn(3, 4)
#' tensor2 = torch_randn(4)
#' torch_matmul(tensor1, tensor2).size()
#' # batched matrix x broadcasted vector
#' tensor1 = torch_randn(10, 3, 4)
#' tensor2 = torch_randn(4)
#' torch_matmul(tensor1, tensor2).size()
#' # batched matrix x batched matrix
#' tensor1 = torch_randn(10, 3, 4)
#' tensor2 = torch_randn(10, 4, 5)
#' torch_matmul(tensor1, tensor2).size()
#' # batched matrix x broadcasted matrix
#' tensor1 = torch_randn(10, 3, 4)
#' tensor2 = torch_randn(4, 5)
#' torch_matmul(tensor1, tensor2).size()
#' }
#'
#' @name torch_matmul
#'
#' @export
NULL


#' Matrix_rank
#'
#' Returns the numerical rank of a 2-D tensor. The method to compute the
#' matrix rank is done using SVD by default. If `symmetric` is ``True``,
#' then `input` is assumed to be symmetric, and the computation of the
#' rank is done by obtaining the eigenvalues.
#' 
#' `tol` is the threshold below which the singular values (or the eigenvalues
#' when `symmetric` is ``True``) are considered to be 0. If `tol` is not
#' specified, `tol` is set to ``S.max() * max(S.size()) * eps`` where `S` is the
#' singular values (or the eigenvalues when `symmetric` is ``True``), and ``eps``
#' is the epsilon value for the datatype of `input`.
#'
#' @section Signatures:
#' 
#' matrix_rank(input, tol=None, symmetric=False) -> Tensor
#'
#'
#' @param input (Tensor) the input 2-D tensor
#' @param tol (float, optional) the tolerance value. Default: ``None``
#' @param symmetric (bool, optional) indicates whether :attr:`input` is symmetric.                               Default: ``False``
#'
#' @examples
#' \dontrun{
#' a = torch_eye(10)
#' torch_matrix_rank(a)
#' b = torch_eye(10)
#' b[0, 0] = 0
#' torch_matrix_rank(b)
#' }
#'
#' @name torch_matrix_rank
#'
#' @export
NULL


#' Matrix_power
#'
#' Returns the matrix raised to the power `n` for square matrices.
#' For batch of matrices, each individual matrix is raised to the power `n`.
#' 
#' If `n` is negative, then the inverse of the matrix (if invertible) is
#' raised to the power `n`.  For a batch of matrices, the batched inverse
#' (if invertible) is raised to the power `n`. If `n` is 0, then an identity matrix
#' is returned.
#'
#' @section Signatures:
#' 
#' matrix_power(input, n) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param n (int) the power to raise the matrix to
#'
#' @examples
#' \dontrun{
#' a = torch_randn(2, 2, 2)
#' a
#' torch_matrix_power(a, 3)
#' }
#'
#' @name torch_matrix_power
#'
#' @export
NULL


#' Max
#'
#' Returns the maximum value of all elements in the `input` tensor.
#'
#' @section Signatures:
#' 
#' max(input) -> Tensor
#'
#'
#' @param {input} NA 
#'
#' @examples
#' \dontrun{
#' a = torch_randn(1, 3)
#' a
#' torch_max(a)
#' }
#'
#' @name torch_max
#'
#' @export
NULL


#' Max
#'
#' Returns a namedtuple ``(values, indices)`` where ``values`` is the maximum
#' value of each row of the `input` tensor in the given dimension
#' `dim`. And ``indices`` is the index location of each maximum value found
#' (argmax).
#' 
#' If `keepdim` is ``True``, the output tensors are of the same size
#' as `input` except in the dimension `dim` where they are of size 1.
#' Otherwise, `dim` is squeezed (see [`torch_squeeze`]), resulting
#' in the output tensors having 1 fewer dimension than `input`.
#'
#' @section Signatures:
#' 
#' max(input, dim, keepdim=False, out=None) -> (Tensor, LongTensor)
#'
#'
#' @param {input} NA 
#' @param {dim} NA 
#' @param {keepdim} NA ``False``.
#' @param out (tuple, optional) the result tuple of two output tensors (max, max_indices)
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4, 4)
#' a
#' torch_max(a, 1)
#' }
#'
#' @name torch_max
#'
#' @export
NULL


#' Max
#'
#' Each element of the tensor `input` is compared with the corresponding
#' element of the tensor `other` and an element-wise maximum is taken.
#' 
#' The shapes of `input` and `other` don't need to match,
#' but they must be :ref:`broadcastable <broadcasting-semantics>`.
#' 
#' \deqn{
#'     \text{out}_i = \max(\text{tensor}_i, \text{other}_i)
#' }
#' @note When the shapes do not match, the shape of the returned output tensor
#'           follows the :ref:`broadcasting rules <broadcasting-semantics>`.
#'
#' @section Signatures:
#' 
#' max(input, other, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param other (Tensor) the second input tensor
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4)
#' a
#' b = torch_randn(4)
#' b
#' torch_max(a, b)
#' }
#'
#' @name torch_max
#'
#' @export
NULL


#' Mean
#'
#' Returns the mean value of all elements in the `input` tensor.
#'
#' @section Signatures:
#' 
#' mean(input) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(1, 3)
#' a
#' torch_mean(a)
#' }
#'
#' @name torch_mean
#'
#' @export
NULL


#' Mean
#'
#' Returns the mean value of each row of the `input` tensor in the given
#' dimension `dim`. If `dim` is a list of dimensions,
#' reduce over all of them.
#' 
#' 
#' If `keepdim` is ``True``, the output tensor is of the same size
#' as `input` except in the dimension(s) `dim` where it is of size 1.
#' Otherwise, `dim` is squeezed (see [`torch_squeeze`]), resulting in the
#' output tensor having 1 (or ``len(dim)``) fewer dimension(s).
#'
#' @section Signatures:
#' 
#' mean(input, dim, keepdim=False, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param dim (int or tuple of ints) the dimension or dimensions to reduce.
#' @param keepdim (bool) whether the output tensor has :attr:`dim` retained or not.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4, 4)
#' a
#' torch_mean(a, 1)
#' torch_mean(a, 1, True)
#' }
#'
#' @name torch_mean
#'
#' @export
NULL


#' Median
#'
#' Returns the median value of all elements in the `input` tensor.
#'
#' @section Signatures:
#' 
#' median(input) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(1, 3)
#' a
#' torch_median(a)
#' }
#'
#' @name torch_median
#'
#' @export
NULL


#' Median
#'
#' Returns a namedtuple ``(values, indices)`` where ``values`` is the median
#' value of each row of the `input` tensor in the given dimension
#' `dim`. And ``indices`` is the index location of each median value found.
#' 
#' By default, `dim` is the last dimension of the `input` tensor.
#' 
#' If `keepdim` is ``True``, the output tensors are of the same size
#' as `input` except in the dimension `dim` where they are of size 1.
#' Otherwise, `dim` is squeezed (see [`torch_squeeze`]), resulting in
#' the outputs tensor having 1 fewer dimension than `input`.
#'
#' @section Signatures:
#' 
#' median(input, dim=-1, keepdim=False, values=None, indices=None) -> (Tensor, LongTensor)
#'
#'
#' @param input (Tensor) the input tensor.
#' @param dim (int) the dimension to reduce.
#' @param keepdim (bool) whether the output tensor has :attr:`dim` retained or not.
#' @param values (Tensor, optional) the output tensor
#' @param indices (Tensor, optional) the output index tensor
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4, 5)
#' a
#' torch_median(a, 1)
#' }
#'
#' @name torch_median
#'
#' @export
NULL


#' Min
#'
#' Returns the minimum value of all elements in the `input` tensor.
#'
#' @section Signatures:
#' 
#' min(input) -> Tensor
#'
#'
#' @param {input} NA 
#'
#' @examples
#' \dontrun{
#' a = torch_randn(1, 3)
#' a
#' torch_min(a)
#' }
#'
#' @name torch_min
#'
#' @export
NULL


#' Min
#'
#' Returns a namedtuple ``(values, indices)`` where ``values`` is the minimum
#' value of each row of the `input` tensor in the given dimension
#' `dim`. And ``indices`` is the index location of each minimum value found
#' (argmin).
#' 
#' If `keepdim` is ``True``, the output tensors are of the same size as
#' `input` except in the dimension `dim` where they are of size 1.
#' Otherwise, `dim` is squeezed (see [`torch_squeeze`]), resulting in
#' the output tensors having 1 fewer dimension than `input`.
#'
#' @section Signatures:
#' 
#' min(input, dim, keepdim=False, out=None) -> (Tensor, LongTensor)
#'
#'
#' @param {input} NA 
#' @param {dim} NA 
#' @param {keepdim} NA 
#' @param out (tuple, optional) the tuple of two output tensors (min, min_indices)
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4, 4)
#' a
#' torch_min(a, 1)
#' }
#'
#' @name torch_min
#'
#' @export
NULL


#' Min
#'
#' Each element of the tensor `input` is compared with the corresponding
#' element of the tensor `other` and an element-wise minimum is taken.
#' The resulting tensor is returned.
#' 
#' The shapes of `input` and `other` don't need to match,
#' but they must be :ref:`broadcastable <broadcasting-semantics>`.
#' 
#' \deqn{
#'     \text{out}_i = \min(\text{tensor}_i, \text{other}_i)
#' }
#' @note When the shapes do not match, the shape of the returned output tensor
#'           follows the :ref:`broadcasting rules <broadcasting-semantics>`.
#'
#' @section Signatures:
#' 
#' min(input, other, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param other (Tensor) the second input tensor
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4)
#' a
#' b = torch_randn(4)
#' b
#' torch_min(a, b)
#' }
#'
#' @name torch_min
#'
#' @export
NULL


#' Mm
#'
#' Performs a matrix multiplication of the matrices `input` and `mat2`.
#' 
#' If `input` is a \eqn{(n \times m)} tensor, `mat2` is a
#' \eqn{(m \times p)} tensor, `out` will be a \eqn{(n \times p)} tensor.
#' 
#' @note This function does not :ref:`broadcast <broadcasting-semantics>`.
#'           For broadcasting matrix products, see [`torch_matmul`].
#'
#' @section Signatures:
#' 
#' mm(input, mat2, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the first matrix to be multiplied
#' @param mat2 (Tensor) the second matrix to be multiplied
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' mat1 = torch_randn(2, 3)
#' mat2 = torch_randn(3, 3)
#' torch_mm(mat1, mat2)
#' }
#'
#' @name torch_mm
#'
#' @export
NULL


#' Mode
#'
#' Returns a namedtuple ``(values, indices)`` where ``values`` is the mode
#' value of each row of the `input` tensor in the given dimension
#' `dim`, i.e. a value which appears most often
#' in that row, and ``indices`` is the index location of each mode value found.
#' 
#' By default, `dim` is the last dimension of the `input` tensor.
#' 
#' If `keepdim` is ``True``, the output tensors are of the same size as
#' `input` except in the dimension `dim` where they are of size 1.
#' Otherwise, `dim` is squeezed (see [`torch_squeeze`]), resulting
#' in the output tensors having 1 fewer dimension than `input`.
#' 
#' @note This function is not defined for ``torch_cuda.Tensor`` yet.
#'
#' @section Signatures:
#' 
#' mode(input, dim=-1, keepdim=False, values=None, indices=None) -> (Tensor, LongTensor)
#'
#'
#' @param input (Tensor) the input tensor.
#' @param dim (int) the dimension to reduce.
#' @param keepdim (bool) whether the output tensor has :attr:`dim` retained or not.
#' @param values (Tensor, optional) the output tensor
#' @param indices (Tensor, optional) the output index tensor
#'
#' @examples
#' \dontrun{
#' a = torch_randint(10, (5,))
#' a
#' b = a + (torch_randn(50, 1) * 5).long()
#' torch_mode(b, 0)
#' }
#'
#' @name torch_mode
#'
#' @export
NULL


#' Mul
#'
#' mul(input, other, out=None)
#' 
#' Multiplies each element of the input `input` with the scalar
#' `other` and returns a new resulting tensor.
#' 
#' \deqn{
#'     \text{out}_i = \text{other} \times \text{input}_i
#' }
#' If `input` is of type `FloatTensor` or `DoubleTensor`, `other`
#' should be a real number, otherwise it should be an integer
#'
#' 
#'
#' @param {input} NA 
#' @param value (Number) the number to be multiplied to each element of :attr:`input`
#' @param {out} NA 
#'
#' @examples
#' \dontrun{
#' a = torch_randn(3)
#' a
#' torch_mul(a, 100)
#' }
#'
#' @name torch_mul
#'
#' @export
NULL


#' Mul
#'
#' mul(input, other, out=None)
#' 
#' Each element of the tensor `input` is multiplied by the corresponding
#' element of the Tensor `other`. The resulting tensor is returned.
#' 
#' The shapes of `input` and `other` must be
#' :ref:`broadcastable <broadcasting-semantics>`.
#' 
#' \deqn{
#'     \text{out}_i = \text{input}_i \times \text{other}_i
#' }
#'
#' 
#'
#' @param input (Tensor) the first multiplicand tensor
#' @param other (Tensor) the second multiplicand tensor
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4, 1)
#' a
#' b = torch_randn(1, 4)
#' b
#' torch_mul(a, b)
#' }
#'
#' @name torch_mul
#'
#' @export
NULL


#' Mv
#'
#' Performs a matrix-vector product of the matrix `input` and the vector
#' `vec`.
#' 
#' If `input` is a \eqn{(n \times m)} tensor, `vec` is a 1-D tensor of
#' size \eqn{m}, `out` will be 1-D of size \eqn{n}.
#' 
#' @note This function does not :ref:`broadcast <broadcasting-semantics>`.
#'
#' @section Signatures:
#' 
#' mv(input, vec, out=None) -> Tensor
#'
#'
#' @param input (Tensor) matrix to be multiplied
#' @param vec (Tensor) vector to be multiplied
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' mat = torch_randn(2, 3)
#' vec = torch_randn(3)
#' torch_mv(mat, vec)
#' }
#'
#' @name torch_mv
#'
#' @export
NULL


#' Mvlgamma
#'
#' Computes the multivariate log-gamma function (`[reference]`_) with dimension \eqn{p} element-wise, given by
#' 
#' \deqn{
#'     \log(\Gamma_{p}(a)) = C + \displaystyle \sum_{i=1}^{p} \log\left(\Gamma\left(a - \frac{i - 1}{2}\right)\right)
#' }
#' where \eqn{C = \log(\pi) \times \frac{p (p - 1)}{4}} and \eqn{\Gamma(\cdot)} is the Gamma function.
#' 
#' If any of the elements are less than or equal to \eqn{\frac{p - 1}{2}}, then an error
#' is thrown.
#'
#' @section Signatures:
#' 
#' mvlgamma(input, p) -> Tensor
#'
#'
#' @param input (Tensor) the tensor to compute the multivariate log-gamma function
#' @param p (int) the number of dimensions
#'
#' @examples
#' \dontrun{
#' a = torch_empty(2, 3).uniform_(1, 2)
#' a
#' torch_mvlgamma(a, 2)
#' }
#'
#' @name torch_mvlgamma
#'
#' @export
NULL


#' Narrow
#'
#' Returns a new tensor that is a narrowed version of `input` tensor. The
#' dimension `dim` is input from `start` to `start + length`. The
#' returned tensor and `input` tensor share the same underlying storage.
#'
#' @section Signatures:
#' 
#' narrow(input, dim, start, length) -> Tensor
#'
#'
#' @param input (Tensor) the tensor to narrow
#' @param dim (int) the dimension along which to narrow
#' @param start (int) the starting dimension
#' @param length (int) the distance to the ending dimension
#'
#' @examples
#' \dontrun{
#' x = torch_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#' torch_narrow(x, 0, 0, 2)
#' torch_narrow(x, 1, 1, 2)
#' }
#'
#' @name torch_narrow
#'
#' @export
NULL


#' Ones
#'
#' Returns a tensor filled with the scalar value `1`, with the shape defined
#' by the variable argument `size`.
#'
#' @section Signatures:
#' 
#' ones(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
#'
#'
#' @param size (int...) a sequence of integers defining the shape of the output tensor.        Can be a variable number of arguments or a collection like a list or tuple.
#' @param out (Tensor, optional) the output tensor.
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of returned tensor.        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
#' @param layout (:class:`torch.layout`, optional) the desired layout of returned Tensor.        Default: ``torch.strided``.
#' @param device (:class:`torch.device`, optional) the desired device of returned tensor.        Default: if ``None``, uses the current device for the default tensor type        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU        for CPU tensor types and the current CUDA device for CUDA tensor types.
#' @param requires_grad (bool, optional) If autograd should record operations on the        returned tensor. Default: ``False``.
#'
#' @examples
#' \dontrun{
#' torch_ones(2, 3)
#' torch_ones(5)
#' }
#'
#' @name torch_ones
#'
#' @export
NULL


#' Ones_like
#'
#' Returns a tensor filled with the scalar value `1`, with the same size as
#' `input`. ``torch_ones_like(input)`` is equivalent to
#' ``torch_ones(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)``.
#' 
#' @section Warning:
#'     As of 0.4, this function does not support an `out` keyword. As an alternative,
#'     the old ``torch_ones_like(input, out=output)`` is equivalent to
#'     ``torch_ones(input.size(), out=output)``.
#'
#' @section Signatures:
#' 
#' ones_like(input, dtype=None, layout=None, device=None, requires_grad=False) -> Tensor
#'
#'
#' @param input (Tensor) the size of :attr:`input` will determine size of the output tensor.
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of returned Tensor.        Default: if ``None``, defaults to the dtype of :attr:`input`.
#' @param layout (:class:`torch.layout`, optional) the desired layout of returned tensor.        Default: if ``None``, defaults to the layout of :attr:`input`.
#' @param device (:class:`torch.device`, optional) the desired device of returned tensor.        Default: if ``None``, defaults to the device of :attr:`input`.
#' @param requires_grad (bool, optional) If autograd should record operations on the        returned tensor. Default: ``False``.
#'
#' @examples
#' \dontrun{
#' input = torch_empty(2, 3)
#' torch_ones_like(input)
#' }
#'
#' @name torch_ones_like
#'
#' @export
NULL


#' Cdist
#'
#' Computes batched the p-norm distance between each pair of the two collections of row vectors.
#'
#' 
#'
#' @param x1 (Tensor) input tensor of shape \eqn{B \times P \times M}.
#' @param x2 (Tensor) input tensor of shape \eqn{B \times R \times M}.
#' @param p NA p value for the p-norm distance to calculate between each vector pair        \eqn{\in [0, \infty]}.
#' @param compute_mode NA 'use_mm_for_euclid_dist_if_necessary' - will use matrix multiplication approach to calculate        euclidean distance (p = 2) if P > 25 or R > 25        'use_mm_for_euclid_dist' - will always use matrix multiplication approach to calculate        euclidean distance (p = 2)        'donot_use_mm_for_euclid_dist' - will never use matrix multiplication approach to calculate        euclidean distance (p = 2)        Default: use_mm_for_euclid_dist_if_necessary.
#'
#' 
#'
#' @name torch_cdist
#'
#' @export
NULL


#' Pdist
#'
#' Computes the p-norm distance between every pair of row vectors in the input.
#' This is identical to the upper triangular portion, excluding the diagonal, of
#' `torch_norm(input[:, None] - input, dim=2, p=p)`. This function will be faster
#' if the rows are contiguous.
#' 
#' If input has shape \eqn{N \times M} then the output will have shape
#' \eqn{\frac{1}{2} N (N - 1)}.
#' 
#' This function is equivalent to `scipy.spatial.distance.pdist(input,
#' 'minkowski', p=p)` if \eqn{p \in (0, \infty)}. When \eqn{p = 0} it is
#' equivalent to `scipy.spatial.distance.pdist(input, 'hamming') * M`.
#' When \eqn{p = \infty}, the closest scipy function is
#' `scipy.spatial.distance.pdist(xn, lambda x, y: np.abs(x - y).max())`.
#'
#' @section Signatures:
#' 
#' pdist(input, p=2) -> Tensor
#'
#'
#' @param input NA input tensor of shape \eqn{N \times M}.
#' @param p NA p value for the p-norm distance to calculate between each vector pair        \eqn{\in [0, \infty]}.
#'
#' 
#'
#' @name torch_pdist
#'
#' @export
NULL


#' Cosine_similarity
#'
#' Returns cosine similarity between x1 and x2, computed along dim.
#' 
#' \deqn{
#'     \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}
#' }
#'
#' @section Signatures:
#' 
#' cosine_similarity(x1, x2, dim=1, eps=1e-8) -> Tensor
#'
#'
#' @param x1 (Tensor) First input.
#' @param x2 (Tensor) Second input (of size matching x1).
#' @param dim (int, optional) Dimension of vectors. Default: 1
#' @param eps (float, optional) Small value to avoid division by zero.        Default: 1e-8
#'
#' @examples
#' \dontrun{
#' input1 = torch_randn(100, 128)
#' input2 = torch_randn(100, 128)
#' output = F.cosine_similarity(input1, input2)
#' print(output)
#' }
#'
#' @name torch_cosine_similarity
#'
#' @export
NULL


#' Pixel_shuffle
#'
#' Rearranges elements in a tensor of shape \eqn{(*, C \times r^2, H, W)} to a
#' tensor of shape \eqn{(*, C, H \times r, W \times r)}.
#' 
#' See :class:`~torch.nn.PixelShuffle` for details.
#'
#' 
#'
#' @param input (Tensor) the input tensor
#' @param upscale_factor (int) factor to increase spatial resolution by
#'
#' 
#'
#' @name torch_pixel_shuffle
#'
#' @export
NULL


#' Pinverse
#'
#' Calculates the pseudo-inverse (also known as the Moore-Penrose inverse) of a 2D tensor.
#' Please look at `Moore-Penrose inverse`_ for more details
#' 
#' @note
#'     This method is implemented using the Singular Value Decomposition.
#' 
#' @note
#'     The pseudo-inverse is not necessarily a continuous function in the elements of the matrix `[1]`_.
#'     Therefore, derivatives are not always existent, and exist for a constant rank only `[2]`_.
#'     However, this method is backprop-able due to the implementation by using SVD results, and
#'     could be unstable. Double-backward will also be unstable due to the usage of SVD internally.
#'     See :meth:`~torch.svd` for more details.
#'
#' @section Signatures:
#' 
#' pinverse(input, rcond=1e-15) -> Tensor
#'
#'
#' @param input (Tensor) The input tensor of size \eqn{(*, m, n)} where \eqn{*} is zero or more batch dimensions
#' @param rcond (float) A floating point value to determine the cutoff for small singular values.                   Default: 1e-15
#'
#' @examples
#' \dontrun{
#' input = torch_randn(3, 5)
#' input
#' torch_pinverse(input)
#' # Batched pinverse example
#' a = torch_randn(2,6,3)
#' b = torch_pinverse(a)
#' torch_matmul(b, a)
#' }
#'
#' @name torch_pinverse
#'
#' @export
NULL


#' Rand
#'
#' Returns a tensor filled with random numbers from a uniform distribution
#' on the interval \eqn{[0, 1)}
#' 
#' The shape of the tensor is defined by the variable argument `size`.
#'
#' @section Signatures:
#' 
#' rand(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
#'
#'
#' @param size (int...) a sequence of integers defining the shape of the output tensor.        Can be a variable number of arguments or a collection like a list or tuple.
#' @param out (Tensor, optional) the output tensor.
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of returned tensor.        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
#' @param layout (:class:`torch.layout`, optional) the desired layout of returned Tensor.        Default: ``torch.strided``.
#' @param device (:class:`torch.device`, optional) the desired device of returned tensor.        Default: if ``None``, uses the current device for the default tensor type        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU        for CPU tensor types and the current CUDA device for CUDA tensor types.
#' @param requires_grad (bool, optional) If autograd should record operations on the        returned tensor. Default: ``False``.
#'
#' @examples
#' \dontrun{
#' torch_rand(4)
#' torch_rand(2, 3)
#' }
#'
#' @name torch_rand
#'
#' @export
NULL


#' Rand_like
#'
#' Returns a tensor with the same size as `input` that is filled with
#' random numbers from a uniform distribution on the interval \eqn{[0, 1)}.
#' ``torch_rand_like(input)`` is equivalent to
#' ``torch_rand(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)``.
#'
#' @section Signatures:
#' 
#' rand_like(input, dtype=None, layout=None, device=None, requires_grad=False) -> Tensor
#'
#'
#' @param input (Tensor) the size of :attr:`input` will determine size of the output tensor.
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of returned Tensor.        Default: if ``None``, defaults to the dtype of :attr:`input`.
#' @param layout (:class:`torch.layout`, optional) the desired layout of returned tensor.        Default: if ``None``, defaults to the layout of :attr:`input`.
#' @param device (:class:`torch.device`, optional) the desired device of returned tensor.        Default: if ``None``, defaults to the device of :attr:`input`.
#' @param requires_grad (bool, optional) If autograd should record operations on the        returned tensor. Default: ``False``.
#'
#' 
#'
#' @name torch_rand_like
#'
#' @export
NULL


#' Randint
#'
#' Returns a tensor filled with random integers generated uniformly
#' between `low` (inclusive) and `high` (exclusive).
#' 
#' The shape of the tensor is defined by the variable argument `size`.
#' 
#' .. note:
#'     With the global dtype default (``torch_float32``), this function returns
#'     a tensor with dtype ``torch_int64``.
#'
#' @section Signatures:
#' 
#' randint(low=0, high, size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
#'
#'
#' @param low (int, optional) Lowest integer to be drawn from the distribution. Default: 0.
#' @param high (int) One above the highest integer to be drawn from the distribution.
#' @param size (tuple) a tuple defining the shape of the output tensor.
#' @param generator (:class:`torch.Generator`, optional) a pseudorandom number generator for sampling
#' @param out (Tensor, optional) the output tensor.
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of returned tensor.        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
#' @param layout (:class:`torch.layout`, optional) the desired layout of returned Tensor.        Default: ``torch.strided``.
#' @param device (:class:`torch.device`, optional) the desired device of returned tensor.        Default: if ``None``, uses the current device for the default tensor type        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU        for CPU tensor types and the current CUDA device for CUDA tensor types.
#' @param requires_grad (bool, optional) If autograd should record operations on the        returned tensor. Default: ``False``.
#'
#' @examples
#' \dontrun{
#' torch_randint(3, 5, (3,))
#' torch_randint(10, (2, 2))
#' torch_randint(3, 10, (2, 2))
#' }
#'
#' @name torch_randint
#'
#' @export
NULL


#' Randint_like
#'
#' Returns a tensor with the same shape as Tensor `input` filled with
#' random integers generated uniformly between `low` (inclusive) and
#' `high` (exclusive).
#' 
#' .. note:
#'     With the global dtype default (``torch_float32``), this function returns
#'     a tensor with dtype ``torch_int64``.
#'
#' @section Signatures:
#' 
#' randint_like(input, low=0, high, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
#'
#'
#' @param input (Tensor) the size of :attr:`input` will determine size of the output tensor.
#' @param low (int, optional) Lowest integer to be drawn from the distribution. Default: 0.
#' @param high (int) One above the highest integer to be drawn from the distribution.
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of returned Tensor.        Default: if ``None``, defaults to the dtype of :attr:`input`.
#' @param layout (:class:`torch.layout`, optional) the desired layout of returned tensor.        Default: if ``None``, defaults to the layout of :attr:`input`.
#' @param device (:class:`torch.device`, optional) the desired device of returned tensor.        Default: if ``None``, defaults to the device of :attr:`input`.
#' @param requires_grad (bool, optional) If autograd should record operations on the        returned tensor. Default: ``False``.
#'
#' 
#'
#' @name torch_randint_like
#'
#' @export
NULL


#' Randn
#'
#' Returns a tensor filled with random numbers from a normal distribution
#' with mean `0` and variance `1` (also called the standard normal
#' distribution).
#' 
#' \deqn{
#'     \text{out}_{i} \sim \mathcal{N}(0, 1)
#' }
#' The shape of the tensor is defined by the variable argument `size`.
#'
#' @section Signatures:
#' 
#' randn(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
#'
#'
#' @param size (int...) a sequence of integers defining the shape of the output tensor.        Can be a variable number of arguments or a collection like a list or tuple.
#' @param out (Tensor, optional) the output tensor.
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of returned tensor.        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
#' @param layout (:class:`torch.layout`, optional) the desired layout of returned Tensor.        Default: ``torch.strided``.
#' @param device (:class:`torch.device`, optional) the desired device of returned tensor.        Default: if ``None``, uses the current device for the default tensor type        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU        for CPU tensor types and the current CUDA device for CUDA tensor types.
#' @param requires_grad (bool, optional) If autograd should record operations on the        returned tensor. Default: ``False``.
#'
#' @examples
#' \dontrun{
#' torch_randn(4)
#' torch_randn(2, 3)
#' }
#'
#' @name torch_randn
#'
#' @export
NULL


#' Randn_like
#'
#' Returns a tensor with the same size as `input` that is filled with
#' random numbers from a normal distribution with mean 0 and variance 1.
#' ``torch_randn_like(input)`` is equivalent to
#' ``torch_randn(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)``.
#'
#' @section Signatures:
#' 
#' randn_like(input, dtype=None, layout=None, device=None, requires_grad=False) -> Tensor
#'
#'
#' @param input (Tensor) the size of :attr:`input` will determine size of the output tensor.
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of returned Tensor.        Default: if ``None``, defaults to the dtype of :attr:`input`.
#' @param layout (:class:`torch.layout`, optional) the desired layout of returned tensor.        Default: if ``None``, defaults to the layout of :attr:`input`.
#' @param device (:class:`torch.device`, optional) the desired device of returned tensor.        Default: if ``None``, defaults to the device of :attr:`input`.
#' @param requires_grad (bool, optional) If autograd should record operations on the        returned tensor. Default: ``False``.
#'
#' 
#'
#' @name torch_randn_like
#'
#' @export
NULL


#' Randperm
#'
#' Returns a random permutation of integers from ``0`` to ``n - 1``.
#'
#' @section Signatures:
#' 
#' randperm(n, out=None, dtype=torch.int64, layout=torch.strided, device=None, requires_grad=False) -> LongTensor
#'
#'
#' @param n (int) the upper bound (exclusive)
#' @param out (Tensor, optional) the output tensor.
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of returned tensor.        Default: ``torch.int64``.
#' @param layout (:class:`torch.layout`, optional) the desired layout of returned Tensor.        Default: ``torch.strided``.
#' @param device (:class:`torch.device`, optional) the desired device of returned tensor.        Default: if ``None``, uses the current device for the default tensor type        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU        for CPU tensor types and the current CUDA device for CUDA tensor types.
#' @param requires_grad (bool, optional) If autograd should record operations on the        returned tensor. Default: ``False``.
#'
#' @examples
#' \dontrun{
#' torch_randperm(4)
#' }
#'
#' @name torch_randperm
#'
#' @export
NULL


#' Range
#'
#' Returns a 1-D tensor of size \eqn{\left\lfloor \frac{\text{end} - \text{start}}{\text{step}} \right\rfloor + 1}
#' with values from `start` to `end` with step `step`. Step is
#' the gap between two values in the tensor.
#' 
#' \deqn{
#'     \text{out}_{i+1} = \text{out}_i + \text{step}.
#' }
#' @section Warning:
#'     This function is deprecated in favor of [`torch_arange`].
#'
#' @section Signatures:
#' 
#' range(start=0, end, step=1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
#'
#'
#' @param start (float) the starting value for the set of points. Default: ``0``.
#' @param end (float) the ending value for the set of points
#' @param step (float) the gap between each pair of adjacent points. Default: ``1``.
#' @param out (Tensor, optional) the output tensor.
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of returned tensor.        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`). If `dtype` is not given, infer the data type from the other input        arguments. If any of `start`, `end`, or `stop` are floating-point, the        `dtype` is inferred to be the default dtype, see        :meth:`~torch.get_default_dtype`. Otherwise, the `dtype` is inferred to        be `torch.int64`.
#' @param layout (:class:`torch.layout`, optional) the desired layout of returned Tensor.        Default: ``torch.strided``.
#' @param device (:class:`torch.device`, optional) the desired device of returned tensor.        Default: if ``None``, uses the current device for the default tensor type        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU        for CPU tensor types and the current CUDA device for CUDA tensor types.
#' @param requires_grad (bool, optional) If autograd should record operations on the        returned tensor. Default: ``False``.
#'
#' @examples
#' \dontrun{
#' torch_range(1, 4)
#' torch_range(1, 4, 0.5)
#' }
#'
#' @name torch_range
#'
#' @export
NULL


#' Reciprocal
#'
#' Returns a new tensor with the reciprocal of the elements of `input`
#' 
#' \deqn{
#'     \text{out}_{i} = \frac{1}{\text{input}_{i}}
#' }
#'
#' @section Signatures:
#' 
#' reciprocal(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4)
#' a
#' torch_reciprocal(a)
#' }
#'
#' @name torch_reciprocal
#'
#' @export
NULL


#' Neg
#'
#' Returns a new tensor with the negative of the elements of `input`.
#' 
#' \deqn{
#'     \text{out} = -1 \times \text{input}
#' }
#'
#' @section Signatures:
#' 
#' neg(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(5)
#' a
#' torch_neg(a)
#' }
#'
#' @name torch_neg
#'
#' @export
NULL


#' Repeat_interleave
#'
#' Repeat elements of a tensor.
#' 
#' @section Warning:
#' 
#'     This is different from [`torch_repeat`] but similar to `numpy.repeat`.
#'
#' @section Signatures:
#' 
#' repeat_interleave(input, repeats, dim=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param repeats (Tensor or int) The number of repetitions for each element.        repeats is broadcasted to fit the shape of the given axis.
#' @param dim (int, optional) The dimension along which to repeat values.        By default, use the flattened input array, and return a flat output        array.
#'
#' @examples
#' \dontrun{
#' x = torch_tensor([1, 2, 3])
#' x.repeat_interleave(2)
#' y = torch_tensor([[1, 2], [3, 4]])
#' torch_repeat_interleave(y, 2)
#' torch_repeat_interleave(y, 3, dim=1)
#' torch_repeat_interleave(y, torch_tensor([1, 2]), dim=0)
#' }
#'
#' @name torch_repeat_interleave
#'
#' @export
NULL


#' Repeat_interleave
#'
#' If the `repeats` is `tensor([n1, n2, n3, ...])`, then the output will be
#' `tensor([0, 0, ..., 1, 1, ..., 2, 2, ..., ...])` where `0` appears `n1` times,
#' `1` appears `n2` times, `2` appears `n3` times, etc.
#'
#' @section Signatures:
#' 
#' repeat_interleave(repeats) -> Tensor
#'
#'
#'
#'
#' 
#'
#' @name torch_repeat_interleave
#'
#' @export
NULL


#' Reshape
#'
#' Returns a tensor with the same data and number of elements as `input`,
#' but with the specified shape. When possible, the returned tensor will be a view
#' of `input`. Otherwise, it will be a copy. Contiguous inputs and inputs
#' with compatible strides can be reshaped without copying, but you should not
#' depend on the copying vs. viewing behavior.
#' 
#' See :meth:`torch_Tensor.view` on when it is possible to return a view.
#' 
#' A single dimension may be -1, in which case it's inferred from the remaining
#' dimensions and the number of elements in `input`.
#'
#' @section Signatures:
#' 
#' reshape(input, shape) -> Tensor
#'
#'
#' @param input (Tensor) the tensor to be reshaped
#' @param shape (tuple of ints) the new shape
#'
#' @examples
#' \dontrun{
#' a = torch_arange(4.)
#' torch_reshape(a, (2, 2))
#' b = torch_tensor([[0, 1], [2, 3]])
#' torch_reshape(b, (-1,))
#' }
#'
#' @name torch_reshape
#'
#' @export
NULL


#' Round
#'
#' Returns a new tensor with each of the elements of `input` rounded
#' to the closest integer.
#'
#' @section Signatures:
#' 
#' round(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4)
#' a
#' torch_round(a)
#' }
#'
#' @name torch_round
#'
#' @export
NULL


#' Rrelu_
#'
#' In-place version of [`~rrelu`].
#'
#' @section Signatures:
#' 
#' rrelu_(input, lower=1./8, upper=1./3, training=False) -> Tensor
#'
#'
#'
#'
#' 
#'
#' @name torch_rrelu_
#'
#' @export
NULL


#' Relu_
#'
#' In-place version of [`~relu`].
#'
#' @section Signatures:
#' 
#' relu_(input) -> Tensor
#'
#'
#'
#'
#' 
#'
#' @name torch_relu_
#'
#' @export
NULL


#' Rsqrt
#'
#' Returns a new tensor with the reciprocal of the square-root of each of
#' the elements of `input`.
#' 
#' \deqn{
#'     \text{out}_{i} = \frac{1}{\sqrt{\text{input}_{i}}}
#' }
#'
#' @section Signatures:
#' 
#' rsqrt(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4)
#' a
#' torch_rsqrt(a)
#' }
#'
#' @name torch_rsqrt
#'
#' @export
NULL


#' Selu_
#'
#' In-place version of [`~selu`].
#'
#' @section Signatures:
#' 
#' selu_(input) -> Tensor
#'
#'
#'
#'
#' 
#'
#' @name torch_selu_
#'
#' @export
NULL


#' Celu_
#'
#' In-place version of [`~celu`].
#'
#' @section Signatures:
#' 
#' celu_(input, alpha=1.) -> Tensor
#'
#'
#'
#'
#' 
#'
#' @name torch_celu_
#'
#' @export
NULL


#' Sigmoid
#'
#' Returns a new tensor with the sigmoid of the elements of `input`.
#' 
#' \deqn{
#'     \text{out}_{i} = \frac{1}{1 + e^{-\text{input}_{i}}}
#' }
#'
#' @section Signatures:
#' 
#' sigmoid(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4)
#' a
#' torch_sigmoid(a)
#' }
#'
#' @name torch_sigmoid
#'
#' @export
NULL


#' Sin
#'
#' Returns a new tensor with the sine of the elements of `input`.
#' 
#' \deqn{
#'     \text{out}_{i} = \sin(\text{input}_{i})
#' }
#'
#' @section Signatures:
#' 
#' sin(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4)
#' a
#' torch_sin(a)
#' }
#'
#' @name torch_sin
#'
#' @export
NULL


#' Sinh
#'
#' Returns a new tensor with the hyperbolic sine of the elements of
#' `input`.
#' 
#' \deqn{
#'     \text{out}_{i} = \sinh(\text{input}_{i})
#' }
#'
#' @section Signatures:
#' 
#' sinh(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4)
#' a
#' torch_sinh(a)
#' }
#'
#' @name torch_sinh
#'
#' @export
NULL


#' Slogdet
#'
#' Calculates the sign and log absolute value of the determinant(s) of a square matrix or batches of square matrices.
#' 
#' @note
#'     If ``input`` has zero determinant, this returns ``(0, -inf)``.
#' 
#' @note
#'     Backward through :meth:`slogdet` internally uses SVD results when `input`
#'     is not invertible. In this case, double backward through :meth:`slogdet`
#'     will be unstable in when `input` doesn't have distinct singular values.
#'     See :meth:`~torch.svd` for details.
#'
#' @section Signatures:
#' 
#' slogdet(input) -> (Tensor, Tensor)
#'
#'
#' @param input (Tensor) the input tensor of size ``(*, n, n)`` where ``*`` is zero or more                batch dimensions.
#'
#' @examples
#' \dontrun{
#' A = torch_randn(3, 3)
#' A
#' torch_det(A)
#' torch_logdet(A)
#' torch_slogdet(A)
#' }
#'
#' @name torch_slogdet
#'
#' @export
NULL


#' Split
#'
#' Splits the tensor into chunks.
#' 
#'     If `split_size_or_sections` is an integer type, then `tensor` will
#'     be split into equally sized chunks (if possible). Last chunk will be smaller if
#'     the tensor size along the given dimension `dim` is not divisible by
#'     `split_size`.
#' 
#'     If `split_size_or_sections` is a list, then `tensor` will be split
#'     into ``len(split_size_or_sections)`` chunks with sizes in `dim` according
#'     to `split_size_or_sections`.
#'
#' 
#'
#' @param tensor (Tensor) tensor to split.
#' @param split_size_or_sections (int) size of a single chunk or        list of sizes for each chunk
#' @param dim (int) dimension along which to split the tensor.
#'
#' 
#'
#' @name torch_split
#'
#' @export
NULL


#' Squeeze
#'
#' Returns a tensor with all the dimensions of `input` of size `1` removed.
#' 
#' For example, if `input` is of shape:
#' \eqn{(A \times 1 \times B \times C \times 1 \times D)} then the `out` tensor
#' will be of shape: \eqn{(A \times B \times C \times D)}.
#' 
#' When `dim` is given, a squeeze operation is done only in the given
#' dimension. If `input` is of shape: \eqn{(A \times 1 \times B)},
#' ``squeeze(input, 0)`` leaves the tensor unchanged, but ``squeeze(input, 1)``
#' will squeeze the tensor to the shape \eqn{(A \times B)}.
#' 
#' @note The returned tensor shares the storage with the input tensor,
#'           so changing the contents of one will change the contents of the other.
#'
#' @section Signatures:
#' 
#' squeeze(input, dim=None, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param dim (int, optional) if given, the input will be squeezed only in           this dimension
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' x = torch_zeros(2, 1, 2, 1, 2)
#' x.size()
#' y = torch_squeeze(x)
#' y.size()
#' y = torch_squeeze(x, 0)
#' y.size()
#' y = torch_squeeze(x, 1)
#' y.size()
#' }
#'
#' @name torch_squeeze
#'
#' @export
NULL


#' Stack
#'
#' Concatenates sequence of tensors along a new dimension.
#' 
#' All tensors need to be of the same size.
#'
#' @section Signatures:
#' 
#' stack(tensors, dim=0, out=None) -> Tensor
#'
#'
#' @param tensors (sequence of Tensors) sequence of tensors to concatenate
#' @param dim (int) dimension to insert. Has to be between 0 and the number        of dimensions of concatenated tensors (inclusive)
#' @param out (Tensor, optional) the output tensor.
#'
#' 
#'
#' @name torch_stack
#'
#' @export
NULL


#' Stft
#'
#' Short-time Fourier transform (STFT).
#' 
#'     Ignoring the optional batch dimension, this method computes the following
#'     expression:
#' 
#' \deqn{
#'         X[m, \omega] = \sum_{k = 0}^{\text{win\_length-1}}%
#'                             \text{window}[k]\ \text{input}[m \times \text{hop\_length} + k]\ %
#'                             \exp\left(- j \frac{2 \pi \cdot \omega k}{\text{win\_length}}\right),
#' }
#'     where \eqn{m} is the index of the sliding window, and \eqn{\omega} is
#'     the frequency that \eqn{0 \leq \omega < \text{n\_fft}}. When
#'     `onesided` is the default value ``True``,
#' 
#'     * `input` must be either a 1-D time sequence or a 2-D batch of time
#'       sequences.
#' 
#'     * If `hop_length` is ``None`` (default), it is treated as equal to
#'       ``floor(n_fft / 4)``.
#' 
#'     * If `win_length` is ``None`` (default), it is treated as equal to
#'       `n_fft`.
#' 
#'     * `window` can be a 1-D tensor of size `win_length`, e.g., from
#'       :meth:`torch_hann_window`. If `window` is ``None`` (default), it is
#'       treated as if having \eqn{1} everywhere in the window. If
#'       \eqn{\text{win\_length} < \text{n\_fft}}, `window` will be padded on
#'       both sides to length `n_fft` before being applied.
#' 
#'     * If `center` is ``True`` (default), `input` will be padded on
#'       both sides so that the \eqn{t}-th frame is centered at time
#'       \eqn{t \times \text{hop\_length}}. Otherwise, the \eqn{t}-th frame
#'       begins at time  \eqn{t \times \text{hop\_length}}.
#' 
#'     * `pad_mode` determines the padding method used on `input` when
#'       `center` is ``True``. See :meth:`torch_nn.functional.pad` for
#'       all available options. Default is ``"reflect"``.
#' 
#'     * If `onesided` is ``True`` (default), only values for \eqn{\omega}
#'       in \eqn{\left[0, 1, 2, \dots, \left\lfloor \frac{\text{n\_fft}}{2} \right\rfloor + 1\right]}
#'       are returned because the real-to-complex Fourier transform satisfies the
#'       conjugate symmetry, i.e., \eqn{X[m, \omega] = X[m, \text{n\_fft} - \omega]^*}.
#' 
#'     * If `normalized` is ``True`` (default is ``False``), the function
#'       returns the normalized STFT results, i.e., multiplied by \eqn{(\text{frame\_length})^{-0.5}}.
#' 
#'     Returns the real and the imaginary parts together as one tensor of size
#'     \eqn{(* \times N \times T \times 2)}, where \eqn{*} is the optional
#'     batch size of `input`, \eqn{N} is the number of frequencies where
#'     STFT is applied, \eqn{T} is the total number of frames used, and each pair
#'     in the last dimension represents a complex number as the real part and the
#'     imaginary part.
#' 
#'     .. warning::
#'       This function changed signature at version 0.4.1. Calling with the
#'       previous signature may cause error or return incorrect result.
#'
#' 
#'
#' @param input (Tensor) the input tensor
#' @param n_fft (int) size of Fourier transform
#' @param hop_length (int, optional) the distance between neighboring sliding window        frames. Default: ``None`` (treated as equal to ``floor(n_fft / 4)``)
#' @param win_length (int, optional) the size of window frame and STFT filter.        Default: ``None``  (treated as equal to :attr:`n_fft`)
#' @param window (Tensor, optional) the optional window function.        Default: ``None`` (treated as window of all \eqn{1} s)
#' @param center (bool, optional) whether to pad :attr:`input` on both sides so        that the \eqn{t}-th frame is centered at time \eqn{t \times \text{hop\_length}}.        Default: ``True``
#' @param pad_mode (string, optional) controls the padding method used when        :attr:`center` is ``True``. Default: ``"reflect"``
#' @param normalized (bool, optional) controls whether to return the normalized STFT results         Default: ``False``
#' @param onesided (bool, optional) controls whether to return half of results to        avoid redundancy Default: ``True``
#'
#' 
#'
#' @name torch_stft
#'
#' @export
NULL


#' Sum
#'
#' Returns the sum of all elements in the `input` tensor.
#'
#' @section Signatures:
#' 
#' sum(input, dtype=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of returned tensor.        If specified, the input tensor is casted to :attr:`dtype` before the operation        is performed. This is useful for preventing data type overflows. Default: None.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(1, 3)
#' a
#' torch_sum(a)
#' }
#'
#' @name torch_sum
#'
#' @export
NULL


#' Sum
#'
#' Returns the sum of each row of the `input` tensor in the given
#' dimension `dim`. If `dim` is a list of dimensions,
#' reduce over all of them.
#' 
#' 
#' If `keepdim` is ``True``, the output tensor is of the same size
#' as `input` except in the dimension(s) `dim` where it is of size 1.
#' Otherwise, `dim` is squeezed (see [`torch_squeeze`]), resulting in the
#' output tensor having 1 (or ``len(dim)``) fewer dimension(s).
#'
#' @section Signatures:
#' 
#' sum(input, dim, keepdim=False, dtype=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param dim (int or tuple of ints) the dimension or dimensions to reduce.
#' @param keepdim (bool) whether the output tensor has :attr:`dim` retained or not.
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of returned tensor.        If specified, the input tensor is casted to :attr:`dtype` before the operation        is performed. This is useful for preventing data type overflows. Default: None.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4, 4)
#' a
#' torch_sum(a, 1)
#' b = torch_arange(4 * 5 * 6).view(4, 5, 6)
#' torch_sum(b, (2, 1))
#' }
#'
#' @name torch_sum
#'
#' @export
NULL


#' Sqrt
#'
#' Returns a new tensor with the square-root of the elements of `input`.
#' 
#' \deqn{
#'     \text{out}_{i} = \sqrt{\text{input}_{i}}
#' }
#'
#' @section Signatures:
#' 
#' sqrt(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4)
#' a
#' torch_sqrt(a)
#' }
#'
#' @name torch_sqrt
#'
#' @export
NULL


#' Std
#'
#' Returns the standard-deviation of all elements in the `input` tensor.
#' 
#' If `unbiased` is ``False``, then the standard-deviation will be calculated
#' via the biased estimator. Otherwise, Bessel's correction will be used.
#'
#' @section Signatures:
#' 
#' std(input, unbiased=True) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param unbiased (bool) whether to use the unbiased estimation or not
#'
#' @examples
#' \dontrun{
#' a = torch_randn(1, 3)
#' a
#' torch_std(a)
#' }
#'
#' @name torch_std
#'
#' @export
NULL


#' Std
#'
#' Returns the standard-deviation of each row of the `input` tensor in the
#' dimension `dim`. If `dim` is a list of dimensions,
#' reduce over all of them.
#' 
#' 
#' If `keepdim` is ``True``, the output tensor is of the same size
#' as `input` except in the dimension(s) `dim` where it is of size 1.
#' Otherwise, `dim` is squeezed (see [`torch_squeeze`]), resulting in the
#' output tensor having 1 (or ``len(dim)``) fewer dimension(s).
#' 
#' 
#' If `unbiased` is ``False``, then the standard-deviation will be calculated
#' via the biased estimator. Otherwise, Bessel's correction will be used.
#'
#' @section Signatures:
#' 
#' std(input, dim, keepdim=False, unbiased=True, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param dim (int or tuple of ints) the dimension or dimensions to reduce.
#' @param keepdim (bool) whether the output tensor has :attr:`dim` retained or not.
#' @param unbiased (bool) whether to use the unbiased estimation or not
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4, 4)
#' a
#' torch_std(a, dim=1)
#' }
#'
#' @name torch_std
#'
#' @export
NULL


#' Std_mean
#'
#' Returns the standard-deviation and mean of all elements in the `input` tensor.
#' 
#' If `unbiased` is ``False``, then the standard-deviation will be calculated
#' via the biased estimator. Otherwise, Bessel's correction will be used.
#'
#' @section Signatures:
#' 
#' std_mean(input, unbiased=True) -> (Tensor, Tensor)
#'
#'
#' @param input (Tensor) the input tensor.
#' @param unbiased (bool) whether to use the unbiased estimation or not
#'
#' @examples
#' \dontrun{
#' a = torch_randn(1, 3)
#' a
#' torch_std_mean(a)
#' }
#'
#' @name torch_std_mean
#'
#' @export
NULL


#' Std_mean
#'
#' Returns the standard-deviation and mean of each row of the `input` tensor in the
#' dimension `dim`. If `dim` is a list of dimensions,
#' reduce over all of them.
#' 
#' 
#' If `keepdim` is ``True``, the output tensor is of the same size
#' as `input` except in the dimension(s) `dim` where it is of size 1.
#' Otherwise, `dim` is squeezed (see [`torch_squeeze`]), resulting in the
#' output tensor having 1 (or ``len(dim)``) fewer dimension(s).
#' 
#' 
#' If `unbiased` is ``False``, then the standard-deviation will be calculated
#' via the biased estimator. Otherwise, Bessel's correction will be used.
#'
#' @section Signatures:
#' 
#' std(input, dim, keepdim=False, unbiased=True) -> (Tensor, Tensor)
#'
#'
#' @param input (Tensor) the input tensor.
#' @param dim (int or tuple of ints) the dimension or dimensions to reduce.
#' @param keepdim (bool) whether the output tensor has :attr:`dim` retained or not.
#' @param unbiased (bool) whether to use the unbiased estimation or not
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4, 4)
#' a
#' torch_std_mean(a, 1)
#' }
#'
#' @name torch_std_mean
#'
#' @export
NULL


#' Prod
#'
#' Returns the product of all elements in the `input` tensor.
#'
#' @section Signatures:
#' 
#' prod(input, dtype=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of returned tensor.        If specified, the input tensor is casted to :attr:`dtype` before the operation        is performed. This is useful for preventing data type overflows. Default: None.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(1, 3)
#' a
#' torch_prod(a)
#' }
#'
#' @name torch_prod
#'
#' @export
NULL


#' Prod
#'
#' Returns the product of each row of the `input` tensor in the given
#' dimension `dim`.
#' 
#' If `keepdim` is ``True``, the output tensor is of the same size
#' as `input` except in the dimension `dim` where it is of size 1.
#' Otherwise, `dim` is squeezed (see [`torch_squeeze`]), resulting in
#' the output tensor having 1 fewer dimension than `input`.
#'
#' @section Signatures:
#' 
#' prod(input, dim, keepdim=False, dtype=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param dim (int) the dimension to reduce.
#' @param keepdim (bool) whether the output tensor has :attr:`dim` retained or not.
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of returned tensor.        If specified, the input tensor is casted to :attr:`dtype` before the operation        is performed. This is useful for preventing data type overflows. Default: None.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4, 2)
#' a
#' torch_prod(a, 1)
#' }
#'
#' @name torch_prod
#'
#' @export
NULL


#' T
#'
#' Expects `input` to be <= 2-D tensor and transposes dimensions 0
#' and 1.
#' 
#' 0-D and 1-D tensors are returned as it is and
#' 2-D tensor can be seen as a short-hand function for ``transpose(input, 0, 1)``.
#'
#' @section Signatures:
#' 
#' t(input) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#'
#' @examples
#' \dontrun{
#' x = torch_randn(())
#' x
#' torch_t(x)
#' x = torch_randn(3)
#' x
#' torch_t(x)
#' x = torch_randn(2, 3)
#' x
#' torch_t(x)
#' }
#'
#' @name torch_t
#'
#' @export
NULL


#' Tan
#'
#' Returns a new tensor with the tangent of the elements of `input`.
#' 
#' \deqn{
#'     \text{out}_{i} = \tan(\text{input}_{i})
#' }
#'
#' @section Signatures:
#' 
#' tan(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4)
#' a
#' torch_tan(a)
#' }
#'
#' @name torch_tan
#'
#' @export
NULL


#' Tanh
#'
#' Returns a new tensor with the hyperbolic tangent of the elements
#' of `input`.
#' 
#' \deqn{
#'     \text{out}_{i} = \tanh(\text{input}_{i})
#' }
#'
#' @section Signatures:
#' 
#' tanh(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4)
#' a
#' torch_tanh(a)
#' }
#'
#' @name torch_tanh
#'
#' @export
NULL


#' Tensordot
#'
#' Returns a contraction of a and b over multiple dimensions.
#' 
#'     `tensordot` implements a generalized matrix product.
#'
#' 
#'
#' @param a (Tensor) Left tensor to contract
#' @param b (Tensor) Right tensor to contract
#' @param dims (int or tuple of two lists of integers) number of dimensions to     contract or explicit lists of dimensions for :attr:`a` and     :attr:`b` respectively
#'
#' 
#'
#' @name torch_tensordot
#'
#' @export
NULL


#' Threshold_
#'
#' In-place version of [`~threshold`].
#'
#' @section Signatures:
#' 
#' threshold_(input, threshold, value) -> Tensor
#'
#'
#'
#'
#' 
#'
#' @name torch_threshold_
#'
#' @export
NULL


#' Transpose
#'
#' Returns a tensor that is a transposed version of `input`.
#' The given dimensions `dim0` and `dim1` are swapped.
#' 
#' The resulting `out` tensor shares it's underlying storage with the
#' `input` tensor, so changing the content of one would change the content
#' of the other.
#'
#' @section Signatures:
#' 
#' transpose(input, dim0, dim1) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param dim0 (int) the first dimension to be transposed
#' @param dim1 (int) the second dimension to be transposed
#'
#' @examples
#' \dontrun{
#' x = torch_randn(2, 3)
#' x
#' torch_transpose(x, 0, 1)
#' }
#'
#' @name torch_transpose
#'
#' @export
NULL


#' Flip
#'
#' Reverse the order of a n-D tensor along given axis in dims.
#'
#' @section Signatures:
#' 
#' flip(input, dims) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param dims (a list or tuple) axis to flip on
#'
#' @examples
#' \dontrun{
#' x = torch_arange(8).view(2, 2, 2)
#' x
#' torch_flip(x, [0, 1])
#' }
#'
#' @name torch_flip
#'
#' @export
NULL


#' Roll
#'
#' Roll the tensor along the given dimension(s). Elements that are shifted beyond the
#' last position are re-introduced at the first position. If a dimension is not
#' specified, the tensor will be flattened before rolling and then restored
#' to the original shape.
#'
#' @section Signatures:
#' 
#' roll(input, shifts, dims=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param shifts (int or tuple of ints) The number of places by which the elements        of the tensor are shifted. If shifts is a tuple, dims must be a tuple of        the same size, and each dimension will be rolled by the corresponding        value
#' @param dims (int or tuple of ints) Axis along which to roll
#'
#' @examples
#' \dontrun{
#' x = torch_tensor([1, 2, 3, 4, 5, 6, 7, 8]).view(4, 2)
#' x
#' torch_roll(x, 1, 0)
#' torch_roll(x, -1, 0)
#' torch_roll(x, shifts=(2, 1), dims=(0, 1))
#' }
#'
#' @name torch_roll
#'
#' @export
NULL


#' Rot90
#'
#' Rotate a n-D tensor by 90 degrees in the plane specified by dims axis.
#' Rotation direction is from the first towards the second axis if k > 0, and from the second towards the first for k < 0.
#'
#' @section Signatures:
#' 
#' rot90(input, k, dims) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param k (int) number of times to rotate
#' @param dims (a list or tuple) axis to rotate
#'
#' @examples
#' \dontrun{
#' x = torch_arange(4).view(2, 2)
#' x
#' torch_rot90(x, 1, [0, 1])
#' x = torch_arange(8).view(2, 2, 2)
#' x
#' torch_rot90(x, 1, [1, 2])
#' }
#'
#' @name torch_rot90
#'
#' @export
NULL


#' Trapz
#'
#' Estimate \eqn{\int y\,dx} along `dim`, using the trapezoid rule.
#'
#' @section Signatures:
#' 
#' trapz(y, x, *, dim=-1) -> Tensor
#'
#'
#' @param y (Tensor) The values of the function to integrate
#' @param x (Tensor) The points at which the function `y` is sampled.        If `x` is not in ascending order, intervals on which it is decreasing        contribute negatively to the estimated integral (i.e., the convention        \eqn{\int_a^b f = -\int_b^a f} is followed).
#' @param dim (int) The dimension along which to integrate.        By default, use the last dimension.
#'
#' @examples
#' \dontrun{
#' y = torch_randn((2, 3))
#' y
#' x = torch_tensor([[1, 3, 4], [1, 2, 3]])
#' torch_trapz(y, x)
#' }
#'
#' @name torch_trapz
#'
#' @export
NULL


#' Trapz
#'
#' As above, but the sample points are spaced uniformly at a distance of `dx`.
#'
#' @section Signatures:
#' 
#' trapz(y, *, dx=1, dim=-1) -> Tensor
#'
#'
#' @param y (Tensor) The values of the function to integrate
#' @param dx (float) The distance between points at which `y` is sampled.
#' @param dim (int) The dimension along which to integrate.        By default, use the last dimension.
#'
#' 
#'
#' @name torch_trapz
#'
#' @export
NULL


#' Trunc
#'
#' Returns a new tensor with the truncated integer values of
#' the elements of `input`.
#'
#' @section Signatures:
#' 
#' trunc(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4)
#' a
#' torch_trunc(a)
#' }
#'
#' @name torch_trunc
#'
#' @export
NULL


#' Unique_consecutive
#'
#' Eliminates all but the first element from every consecutive group of equivalent elements.
#' 
#'     .. note:: This function is different from [`torch_unique`] in the sense that this function
#'         only eliminates consecutive duplicate values. This semantics is similar to `std::unique`
#'         in C++.
#'
#' 
#'
#' @param input (Tensor) the input tensor
#' @param return_inverse (bool) Whether to also return the indices for where        elements in the original input ended up in the returned unique list.
#' @param return_counts (bool) Whether to also return the counts for each unique        element.
#' @param dim (int) the dimension to apply unique. If ``None``, the unique of the        flattened input is returned. default: ``None``
#'
#' 
#'
#' @name torch_unique_consecutive
#'
#' @export
NULL


#' Unsqueeze
#'
#' Returns a new tensor with a dimension of size one inserted at the
#' specified position.
#' 
#' The returned tensor shares the same underlying data with this tensor.
#' 
#' A `dim` value within the range ``[-input.dim() - 1, input.dim() + 1)``
#' can be used. Negative `dim` will correspond to :meth:`unsqueeze`
#' applied at `dim` = ``dim + input.dim() + 1``.
#'
#' @section Signatures:
#' 
#' unsqueeze(input, dim, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param dim (int) the index at which to insert the singleton dimension
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' x = torch_tensor([1, 2, 3, 4])
#' torch_unsqueeze(x, 0)
#' torch_unsqueeze(x, 1)
#' }
#'
#' @name torch_unsqueeze
#'
#' @export
NULL


#' Var
#'
#' Returns the variance of all elements in the `input` tensor.
#' 
#' If `unbiased` is ``False``, then the variance will be calculated via the
#' biased estimator. Otherwise, Bessel's correction will be used.
#'
#' @section Signatures:
#' 
#' var(input, unbiased=True) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param unbiased (bool) whether to use the unbiased estimation or not
#'
#' @examples
#' \dontrun{
#' a = torch_randn(1, 3)
#' a
#' torch_var(a)
#' }
#'
#' @name torch_var
#'
#' @export
NULL


#' Var
#'
#' Returns the variance of each row of the `input` tensor in the given
#' dimension `dim`.
#' 
#' 
#' If `keepdim` is ``True``, the output tensor is of the same size
#' as `input` except in the dimension(s) `dim` where it is of size 1.
#' Otherwise, `dim` is squeezed (see [`torch_squeeze`]), resulting in the
#' output tensor having 1 (or ``len(dim)``) fewer dimension(s).
#' 
#' 
#' If `unbiased` is ``False``, then the variance will be calculated via the
#' biased estimator. Otherwise, Bessel's correction will be used.
#'
#' @section Signatures:
#' 
#' var(input, dim, keepdim=False, unbiased=True, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param dim (int or tuple of ints) the dimension or dimensions to reduce.
#' @param keepdim (bool) whether the output tensor has :attr:`dim` retained or not.
#' @param unbiased (bool) whether to use the unbiased estimation or not
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4, 4)
#' a
#' torch_var(a, 1)
#' }
#'
#' @name torch_var
#'
#' @export
NULL


#' Var_mean
#'
#' Returns the variance and mean of all elements in the `input` tensor.
#' 
#' If `unbiased` is ``False``, then the variance will be calculated via the
#' biased estimator. Otherwise, Bessel's correction will be used.
#'
#' @section Signatures:
#' 
#' var_mean(input, unbiased=True) -> (Tensor, Tensor)
#'
#'
#' @param input (Tensor) the input tensor.
#' @param unbiased (bool) whether to use the unbiased estimation or not
#'
#' @examples
#' \dontrun{
#' a = torch_randn(1, 3)
#' a
#' torch_var_mean(a)
#' }
#'
#' @name torch_var_mean
#'
#' @export
NULL


#' Var_mean
#'
#' Returns the variance and mean of each row of the `input` tensor in the given
#' dimension `dim`.
#' 
#' 
#' If `keepdim` is ``True``, the output tensor is of the same size
#' as `input` except in the dimension(s) `dim` where it is of size 1.
#' Otherwise, `dim` is squeezed (see [`torch_squeeze`]), resulting in the
#' output tensor having 1 (or ``len(dim)``) fewer dimension(s).
#' 
#' 
#' If `unbiased` is ``False``, then the variance will be calculated via the
#' biased estimator. Otherwise, Bessel's correction will be used.
#'
#' @section Signatures:
#' 
#' var_mean(input, dim, keepdim=False, unbiased=True) -> (Tensor, Tensor)
#'
#'
#' @param input (Tensor) the input tensor.
#' @param dim (int or tuple of ints) the dimension or dimensions to reduce.
#' @param keepdim (bool) whether the output tensor has :attr:`dim` retained or not.
#' @param unbiased (bool) whether to use the unbiased estimation or not
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4, 4)
#' a
#' torch_var_mean(a, 1)
#' }
#'
#' @name torch_var_mean
#'
#' @export
NULL


#' Where
#'
#' Return a tensor of elements selected from either `x` or `y`, depending on `condition`.
#' 
#' The operation is defined as:
#' 
#' \deqn{
#'     \text{out}_i = \begin{cases}
#'         \text{x}_i & \text{if } \text{condition}_i \\
#'         \text{y}_i & \text{otherwise} \\
#'     \end{cases}
#' }
#' @note
#'     The tensors `condition`, `x`, `y` must be :ref:`broadcastable <broadcasting-semantics>`.
#'
#' @section Signatures:
#' 
#' where(condition, x, y) -> Tensor
#'
#'
#' @param condition (BoolTensor) When True (nonzero), yield x, otherwise yield y
#' @param x (Tensor) values selected at indices where :attr:`condition` is ``True``
#' @param y (Tensor) values selected at indices where :attr:`condition` is ``False``
#'
#' @examples
#' \dontrun{
#' x = torch_randn(3, 2)
#' y = torch_ones(3, 2)
#' x
#' torch_where(x > 0, x, y)
#' }
#'
#' @name torch_where
#'
#' @export
NULL


#' Where
#'
#' ``torch_where(condition)`` is identical to
#' ``torch_nonzero(condition, as_tuple=True)``.
#' 
#' @note
#'     See also [`torch_nonzero`].
#'
#' @section Signatures:
#' 
#' where(condition) -> tuple of LongTensor
#'
#'
#'
#'
#' 
#'
#' @name torch_where
#'
#' @export
NULL


#' Zeros
#'
#' Returns a tensor filled with the scalar value `0`, with the shape defined
#' by the variable argument `size`.
#'
#' @section Signatures:
#' 
#' zeros(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
#'
#'
#' @param size (int...) a sequence of integers defining the shape of the output tensor.        Can be a variable number of arguments or a collection like a list or tuple.
#' @param out (Tensor, optional) the output tensor.
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of returned tensor.        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
#' @param layout (:class:`torch.layout`, optional) the desired layout of returned Tensor.        Default: ``torch.strided``.
#' @param device (:class:`torch.device`, optional) the desired device of returned tensor.        Default: if ``None``, uses the current device for the default tensor type        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU        for CPU tensor types and the current CUDA device for CUDA tensor types.
#' @param requires_grad (bool, optional) If autograd should record operations on the        returned tensor. Default: ``False``.
#'
#' @examples
#' \dontrun{
#' torch_zeros(2, 3)
#' torch_zeros(5)
#' }
#'
#' @name torch_zeros
#'
#' @export
NULL


#' Zeros_like
#'
#' Returns a tensor filled with the scalar value `0`, with the same size as
#' `input`. ``torch_zeros_like(input)`` is equivalent to
#' ``torch_zeros(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)``.
#' 
#' @section Warning:
#'     As of 0.4, this function does not support an `out` keyword. As an alternative,
#'     the old ``torch_zeros_like(input, out=output)`` is equivalent to
#'     ``torch_zeros(input.size(), out=output)``.
#'
#' @section Signatures:
#' 
#' zeros_like(input, dtype=None, layout=None, device=None, requires_grad=False) -> Tensor
#'
#'
#' @param input (Tensor) the size of :attr:`input` will determine size of the output tensor.
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of returned Tensor.        Default: if ``None``, defaults to the dtype of :attr:`input`.
#' @param layout (:class:`torch.layout`, optional) the desired layout of returned tensor.        Default: if ``None``, defaults to the layout of :attr:`input`.
#' @param device (:class:`torch.device`, optional) the desired device of returned tensor.        Default: if ``None``, defaults to the device of :attr:`input`.
#' @param requires_grad (bool, optional) If autograd should record operations on the        returned tensor. Default: ``False``.
#'
#' @examples
#' \dontrun{
#' input = torch_empty(2, 3)
#' torch_zeros_like(input)
#' }
#'
#' @name torch_zeros_like
#'
#' @export
NULL


#' Norm
#'
#' Returns the matrix norm or vector norm of a given tensor.
#'
#' 
#'
#' @param input (Tensor) the input tensor
#' @param p (int, float, inf, -inf, 'fro', 'nuc', optional) the order of norm. Default: ``'fro'``        The following norms can be calculated:        =====  ============================  ==========================        ord    matrix norm                   vector norm        =====  ============================  ==========================        None   Frobenius norm                2-norm        'fro'  Frobenius norm                --        'nuc'  nuclear norm                  --        Other  as vec norm when dim is None  sum(abs(x)**ord)**(1./ord)        =====  ============================  ==========================
#' @param dim (int, 2-tuple of ints, 2-list of ints, optional) If it is an int,        vector norm will be calculated, if it is 2-tuple of ints, matrix norm        will be calculated. If the value is None, matrix norm will be calculated        when the input tensor only has two dimensions, vector norm will be        calculated when the input tensor only has one dimension. If the input        tensor has more than two dimensions, the vector norm will be applied to        last dimension.
#' @param keepdim (bool, optional) whether the output tensors have :attr:`dim`        retained or not. Ignored if :attr:`dim` = ``None`` and        :attr:`out` = ``None``. Default: ``False``
#' @param out (Tensor, optional) the output tensor. Ignored if        :attr:`dim` = ``None`` and :attr:`out` = ``None``.
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of        returned tensor. If specified, the input tensor is casted to        :attr:'dtype' while performing the operation. Default: None.
#'
#' 
#'
#' @name torch_norm
#'
#' @export
NULL


#' Pow
#'
#' Takes the power of each element in `input` with `exponent` and
#' returns a tensor with the result.
#' 
#' `exponent` can be either a single ``float`` number or a `Tensor`
#' with the same number of elements as `input`.
#' 
#' When `exponent` is a scalar value, the operation applied is:
#' 
#' \deqn{
#'     \text{out}_i = x_i ^ \text{exponent}
#' }
#' When `exponent` is a tensor, the operation applied is:
#' 
#' \deqn{
#'     \text{out}_i = x_i ^ {\text{exponent}_i}
#' }
#' When `exponent` is a tensor, the shapes of `input`
#' and `exponent` must be :ref:`broadcastable <broadcasting-semantics>`.
#'
#' @section Signatures:
#' 
#' pow(input, exponent, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param exponent (float or tensor) the exponent value
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4)
#' a
#' torch_pow(a, 2)
#' exp = torch_arange(1., 5.)
#' a = torch_arange(1., 5.)
#' a
#' exp
#' torch_pow(a, exp)
#' }
#'
#' @name torch_pow
#'
#' @export
NULL


#' Pow
#'
#' `self` is a scalar ``float`` value, and `exponent` is a tensor.
#' The returned tensor `out` is of the same shape as `exponent`
#' 
#' The operation applied is:
#' 
#' \deqn{
#'     \text{out}_i = \text{self} ^ {\text{exponent}_i}
#' }
#'
#' @section Signatures:
#' 
#' pow(self, exponent, out=None) -> Tensor
#'
#'
#' @param self (float) the scalar base value for the power operation
#' @param exponent (Tensor) the exponent tensor
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' exp = torch_arange(1., 5.)
#' base = 2
#' torch_pow(base, exp)
#' }
#'
#' @name torch_pow
#'
#' @export
NULL


#' Addmm
#'
#' Performs a matrix multiplication of the matrices `mat1` and `mat2`.
#' The matrix `input` is added to the final result.
#' 
#' If `mat1` is a \eqn{(n \times m)} tensor, `mat2` is a
#' \eqn{(m \times p)} tensor, then `input` must be
#' :ref:`broadcastable <broadcasting-semantics>` with a \eqn{(n \times p)} tensor
#' and `out` will be a \eqn{(n \times p)} tensor.
#' 
#' `alpha` and `beta` are scaling factors on matrix-vector product between
#' `mat1` and `mat2` and the added matrix `input` respectively.
#' 
#' \deqn{
#'     \text{out} = \beta\ \text{input} + \alpha\ (\text{mat1}_i \mathbin{@} \text{mat2}_i)
#' }
#' For inputs of type `FloatTensor` or `DoubleTensor`, arguments `beta` and
#' `alpha` must be real numbers, otherwise they should be integers.
#'
#' @section Signatures:
#' 
#' addmm(beta=1, input, alpha=1, mat1, mat2, out=None) -> Tensor
#'
#'
#' @param beta (Number, optional) multiplier for :attr:`input` (\eqn{\beta})
#' @param input (Tensor) matrix to be added
#' @param alpha (Number, optional) multiplier for \eqn{mat1 @ mat2} (\eqn{\alpha})
#' @param mat1 (Tensor) the first matrix to be multiplied
#' @param mat2 (Tensor) the second matrix to be multiplied
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' M = torch_randn(2, 3)
#' mat1 = torch_randn(2, 3)
#' mat2 = torch_randn(3, 3)
#' torch_addmm(M, mat1, mat2)
#' }
#'
#' @name torch_addmm
#'
#' @export
NULL


#' Sparse_coo_tensor
#'
#' Constructs a sparse tensors in COO(rdinate) format with non-zero elements at the given `indices`
#' with the given `values`. A sparse tensor can be `uncoalesced`, in that case, there are duplicate
#' coordinates in the indices, and the value at that index is the sum of all duplicate value entries:
#' `torch_sparse`_.
#'
#' @section Signatures:
#' 
#' sparse_coo_tensor(indices, values, size=None, dtype=None, device=None, requires_grad=False) -> Tensor
#'
#'
#' @param indices (array_like) Initial data for the tensor. Can be a list, tuple,        NumPy ``ndarray``, scalar, and other types. Will be cast to a :class:`torch.LongTensor`        internally. The indices are the coordinates of the non-zero values in the matrix, and thus        should be two-dimensional where the first dimension is the number of tensor dimensions and        the second dimension is the number of non-zero values.
#' @param values (array_like) Initial values for the tensor. Can be a list, tuple,        NumPy ``ndarray``, scalar, and other types.
#' @param size (list, tuple, or :class:`torch.Size`, optional) Size of the sparse tensor. If not        provided the size will be inferred as the minimum size big enough to hold all non-zero        elements.
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of returned tensor.        Default: if None, infers data type from :attr:`values`.
#' @param device (:class:`torch.device`, optional) the desired device of returned tensor.        Default: if None, uses the current device for the default tensor type        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU        for CPU tensor types and the current CUDA device for CUDA tensor types.
#' @param requires_grad (bool, optional) If autograd should record operations on the        returned tensor. Default: ``False``.
#'
#' @examples
#' \dontrun{
#' i = torch_tensor([[0, 1, 1],
#' v = torch_tensor([3, 4, 5], dtype=torch_float32)
#' torch_sparse_coo_tensor(i, v, [2, 4])
#' torch_sparse_coo_tensor(i, v)  # Shape inference
#' torch_sparse_coo_tensor(i, v, [2, 4],
#' S = torch_sparse_coo_tensor(torch_empty([1, 0]), [], [1])
#' S = torch_sparse_coo_tensor(torch_empty([1, 0]), torch_empty([0, 2]), [1, 2])
#' }
#'
#' @name torch_sparse_coo_tensor
#'
#' @export
NULL


#' Unbind
#'
#' Removes a tensor dimension.
#' 
#' Returns a tuple of all slices along a given dimension, already without it.
#'
#' @section Signatures:
#' 
#' unbind(input, dim=0) -> seq
#'
#'
#' @param input (Tensor) the tensor to unbind
#' @param dim (int) dimension to remove
#'
#' @examples
#' \dontrun{
#' torch_unbind(torch_tensor([[1, 2, 3],
#'                            [4, 5, 6],
#'                            [7, 8, 9]]))
#' }
#'
#' @name torch_unbind
#'
#' @export
NULL


#' Quantize_per_tensor
#'
#' Converts a float tensor to quantized tensor with given scale and zero point.
#'
#' @section Signatures:
#' 
#' quantize_per_tensor(input, scale, zero_point, dtype) -> Tensor
#'
#'
#' @param input (Tensor) float tensor to quantize
#' @param scale (float) scale to apply in quantization formula
#' @param zero_point (int) offset in integer value that maps to float zero
#' @param dtype (:class:`torch.dtype`) the desired data type of returned tensor.        Has to be one of the quantized dtypes: ``torch.quint8``, ``torch.qint8``, ``torch.qint32``
#'
#' @examples
#' \dontrun{
#' torch_quantize_per_tensor(torch_tensor([-1.0, 0.0, 1.0, 2.0]), 0.1, 10, torch_quint8)
#' torch_quantize_per_tensor(torch_tensor([-1.0, 0.0, 1.0, 2.0]), 0.1, 10, torch_quint8).int_repr()
#' }
#'
#' @name torch_quantize_per_tensor
#'
#' @export
NULL


#' Quantize_per_channel
#'
#' Converts a float tensor to per-channel quantized tensor with given scales and zero points.
#'
#' @section Signatures:
#' 
#' quantize_per_channel(input, scales, zero_points, axis, dtype) -> Tensor
#'
#'
#' @param input (Tensor) float tensor to quantize
#' @param scales (Tensor) float 1D tensor of scales to use, size should match ``input.size(axis)``
#' @param zero_points (int) integer 1D tensor of offset to use, size should match ``input.size(axis)``
#' @param axis (int) dimension on which apply per-channel quantization
#' @param dtype (:class:`torch.dtype`) the desired data type of returned tensor.        Has to be one of the quantized dtypes: ``torch.quint8``, ``torch.qint8``, ``torch.qint32``
#'
#' @examples
#' \dontrun{
#' x = torch_tensor([[-1.0, 0.0], [1.0, 2.0]])
#' torch_quantize_per_channel(x, torch_tensor([0.1, 0.01]), torch_tensor([10, 0]), 0, torch_quint8)
#' torch_quantize_per_channel(x, torch_tensor([0.1, 0.01]), torch_tensor([10, 0]), 0, torch_quint8).int_repr()
#' }
#'
#' @name torch_quantize_per_channel
#'
#' @export
NULL


#' Meshgrid
#'
#' Take \eqn{N} tensors, each of which can be either scalar or 1-dimensional
#' vector, and create \eqn{N} N-dimensional grids, where the \eqn{i} :sup:`th` grid is defined by
#' expanding the \eqn{i} :sup:`th` input over dimensions defined by other inputs.
#'
#' 
#'
#' @param tensors (list of Tensor) list of scalars or 1 dimensional tensors. Scalars will be
#' @param treated (1,) 
#'
#' 
#'
#' @name torch_meshgrid
#'
#' @export
NULL


#' Cartesian_prod
#'
#' Do cartesian product of the given sequence of tensors. The behavior is similar to
#'     python's `itertools.product`.
#'
#' 
#'
#' @param *tensors NA any number of 1 dimensional tensors.
#'
#' 
#'
#' @name torch_cartesian_prod
#'
#' @export
NULL


#' Combinations
#'
#' Compute combinations of length \eqn{r} of the given tensor. The behavior is similar to
#' python's `itertools.combinations` when `with_replacement` is set to `False`, and
#' `itertools.combinations_with_replacement` when `with_replacement` is set to `True`.
#'
#' @section Signatures:
#' 
#' combinations(input, r=2, with_replacement=False) -> seq
#'
#'
#' @param input (Tensor) 1D vector.
#' @param r (int, optional) number of elements to combine
#' @param with_replacement (boolean, optional) whether to allow duplication in combination
#'
#' @examples
#' \dontrun{
#' a = [1, 2, 3]
#' list(itertools.combinations(a, r=2))
#' list(itertools.combinations(a, r=3))
#' list(itertools.combinations_with_replacement(a, r=2))
#' tensor_a = torch_tensor(a)
#' torch_combinations(tensor_a)
#' torch_combinations(tensor_a, r=3)
#' torch_combinations(tensor_a, with_replacement=True)
#' }
#'
#' @name torch_combinations
#'
#' @export
NULL


#' Result_type
#'
#' Returns the :class:`torch_dtype` that would result from performing an arithmetic
#' operation on the provided input tensors. See type promotion :ref:`documentation <type-promotion-doc>`
#' for more information on the type promotion logic.
#'
#' @section Signatures:
#' 
#' result_type(tensor1, tensor2) -> dtype
#'
#'
#' @param tensor1 (Tensor or Number) an input tensor or number
#' @param tensor2 (Tensor or Number) an input tensor or number
#'
#' @examples
#' \dontrun{
#' torch_result_type(torch_tensor([1, 2], dtype=torch_int), 1.0)
#' torch_result_type(torch_tensor([1, 2], dtype=torch_uint8), torch_tensor(1))
#' }
#'
#' @name torch_result_type
#'
#' @export
NULL


#' Can_cast
#'
#' Determines if a type conversion is allowed under PyTorch casting rules
#' described in the type promotion :ref:`documentation <type-promotion-doc>`.
#'
#' @section Signatures:
#' 
#' can_cast(from, to) -> bool
#'
#'
#' @param from (dtype) The original :class:`torch.dtype`.
#' @param to (dtype) The target :class:`torch.dtype`.
#'
#' @examples
#' \dontrun{
#' torch_can_cast(torch_double, torch_float)
#' torch_can_cast(torch_float, torch_int)
#' }
#'
#' @name torch_can_cast
#'
#' @export
NULL


#' Promote_types
#'
#' Returns the :class:`torch_dtype` with the smallest size and scalar kind that is
#' not smaller nor of lower kind than either `type1` or `type2`. See type promotion
#' :ref:`documentation <type-promotion-doc>` for more information on the type
#' promotion logic.
#'
#' @section Signatures:
#' 
#' promote_types(type1, type2) -> dtype
#'
#'
#' @param type1 (:class:`torch.dtype`) 
#' @param type2 (:class:`torch.dtype`) 
#'
#' @examples
#' \dontrun{
#' torch_promote_types(torch_int32, torch_float32))
#' torch_promote_types(torch_uint8, torch_long)
#' }
#'
#' @name torch_promote_types
#'
#' @export
NULL


#' Bitwise_xor
#'
#' Computes the bitwise XOR of `input` and `other`. The input tensor must be of
#' integral or Boolean types. For bool tensors, it computes the logical XOR.
#'
#' @section Signatures:
#' 
#' bitwise_xor(input, other, out=None) -> Tensor
#'
#'
#' @param input NA the first input tensor
#' @param other NA the second input tensor
#' @param out (Tensor, optional) the output tensor.
#'
#' 
#'
#' @name torch_bitwise_xor
#'
#' @export
NULL


#' Addbmm
#'
#' Performs a batch matrix-matrix product of matrices stored
#' in `batch1` and `batch2`,
#' with a reduced add step (all matrix multiplications get accumulated
#' along the first dimension).
#' `input` is added to the final result.
#' 
#' `batch1` and `batch2` must be 3-D tensors each containing the
#' same number of matrices.
#' 
#' If `batch1` is a \eqn{(b \times n \times m)} tensor, `batch2` is a
#' \eqn{(b \times m \times p)} tensor, `input` must be
#' :ref:`broadcastable <broadcasting-semantics>` with a \eqn{(n \times p)} tensor
#' and `out` will be a \eqn{(n \times p)} tensor.
#' 
#' \deqn{
#'     out = \beta\ \text{input} + \alpha\ (\sum_{i=0}^{b-1} \text{batch1}_i \mathbin{@} \text{batch2}_i)
#' }
#' For inputs of type `FloatTensor` or `DoubleTensor`, arguments `beta` and `alpha`
#' must be real numbers, otherwise they should be integers.
#'
#' @section Signatures:
#' 
#' addbmm(beta=1, input, alpha=1, batch1, batch2, out=None) -> Tensor
#'
#'
#' @param beta (Number, optional) multiplier for :attr:`input` (\eqn{\beta})
#' @param input (Tensor) matrix to be added
#' @param alpha (Number, optional) multiplier for `batch1 @ batch2` (\eqn{\alpha})
#' @param batch1 (Tensor) the first batch of matrices to be multiplied
#' @param batch2 (Tensor) the second batch of matrices to be multiplied
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' M = torch_randn(3, 5)
#' batch1 = torch_randn(10, 3, 4)
#' batch2 = torch_randn(10, 4, 5)
#' torch_addbmm(M, batch1, batch2)
#' }
#'
#' @name torch_addbmm
#'
#' @export
NULL


#' Diag
#'
#' - If `input` is a vector (1-D tensor), then returns a 2-D square tensor
#'   with the elements of `input` as the diagonal.
#' - If `input` is a matrix (2-D tensor), then returns a 1-D tensor with
#'   the diagonal elements of `input`.
#' 
#' The argument `diagonal` controls which diagonal to consider:
#' 
#' - If `diagonal` = 0, it is the main diagonal.
#' - If `diagonal` > 0, it is above the main diagonal.
#' - If `diagonal` < 0, it is below the main diagonal.
#'
#' @section Signatures:
#' 
#' diag(input, diagonal=0, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param diagonal (int, optional) the diagonal to consider
#' @param out (Tensor, optional) the output tensor.
#'
#' 
#'
#' @name torch_diag
#'
#' @export
NULL


#' Cross
#'
#' Returns the cross product of vectors in dimension `dim` of `input`
#' and `other`.
#' 
#' `input` and `other` must have the same size, and the size of their
#' `dim` dimension should be 3.
#' 
#' If `dim` is not given, it defaults to the first dimension found with the
#' size 3.
#'
#' @section Signatures:
#' 
#' cross(input, other, dim=-1, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param other (Tensor) the second input tensor
#' @param dim (int, optional) the dimension to take the cross-product in.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4, 3)
#' a
#' b = torch_randn(4, 3)
#' b
#' torch_cross(a, b, dim=1)
#' torch_cross(a, b)
#' }
#'
#' @name torch_cross
#'
#' @export
NULL


#' Triu
#'
#' Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices
#' `input`, the other elements of the result tensor `out` are set to 0.
#' 
#' The upper triangular part of the matrix is defined as the elements on and
#' above the diagonal.
#' 
#' The argument `diagonal` controls which diagonal to consider. If
#' `diagonal` = 0, all elements on and above the main diagonal are
#' retained. A positive value excludes just as many diagonals above the main
#' diagonal, and similarly a negative value includes just as many diagonals below
#' the main diagonal. The main diagonal are the set of indices
#' \eqn{\lbrace (i, i) \rbrace} for \eqn{i \in [0, \min\{d_{1}, d_{2}\} - 1]} where
#' \eqn{d_{1}, d_{2}} are the dimensions of the matrix.
#'
#' @section Signatures:
#' 
#' triu(input, diagonal=0, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param diagonal (int, optional) the diagonal to consider
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(3, 3)
#' a
#' torch_triu(a)
#' torch_triu(a, diagonal=1)
#' torch_triu(a, diagonal=-1)
#' b = torch_randn(4, 6)
#' b
#' torch_triu(b, diagonal=1)
#' torch_triu(b, diagonal=-1)
#' }
#'
#' @name torch_triu
#'
#' @export
NULL


#' Tril
#'
#' Returns the lower triangular part of the matrix (2-D tensor) or batch of matrices
#' `input`, the other elements of the result tensor `out` are set to 0.
#' 
#' The lower triangular part of the matrix is defined as the elements on and
#' below the diagonal.
#' 
#' The argument `diagonal` controls which diagonal to consider. If
#' `diagonal` = 0, all elements on and below the main diagonal are
#' retained. A positive value includes just as many diagonals above the main
#' diagonal, and similarly a negative value excludes just as many diagonals below
#' the main diagonal. The main diagonal are the set of indices
#' \eqn{\lbrace (i, i) \rbrace} for \eqn{i \in [0, \min\{d_{1}, d_{2}\} - 1]} where
#' \eqn{d_{1}, d_{2}} are the dimensions of the matrix.
#'
#' @section Signatures:
#' 
#' tril(input, diagonal=0, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param diagonal (int, optional) the diagonal to consider
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(3, 3)
#' a
#' torch_tril(a)
#' b = torch_randn(4, 6)
#' b
#' torch_tril(b, diagonal=1)
#' torch_tril(b, diagonal=-1)
#' }
#'
#' @name torch_tril
#'
#' @export
NULL


#' Tril_indices
#'
#' Returns the indices of the lower triangular part of a `row`-by-
#' `col` matrix in a 2-by-N Tensor, where the first row contains row
#' coordinates of all indices and the second row contains column coordinates.
#' Indices are ordered based on rows and then columns.
#' 
#' The lower triangular part of the matrix is defined as the elements on and
#' below the diagonal.
#' 
#' The argument `offset` controls which diagonal to consider. If
#' `offset` = 0, all elements on and below the main diagonal are
#' retained. A positive value includes just as many diagonals above the main
#' diagonal, and similarly a negative value excludes just as many diagonals below
#' the main diagonal. The main diagonal are the set of indices
#' \eqn{\lbrace (i, i) \rbrace} for \eqn{i \in [0, \min\{d_{1}, d_{2}\} - 1]}
#' where \eqn{d_{1}, d_{2}} are the dimensions of the matrix.
#' 
#' NOTE: when running on 'cuda', row * col must be less than \eqn{2^{59}} to
#' prevent overflow during calculation.
#'
#' @section Signatures:
#' 
#' tril_indices(row, col, offset=0, dtype=torch.long, device='cpu', layout=torch.strided) -> Tensor
#'
#'
#' @param row (``int``) number of rows in the 2-D matrix.
#' @param col (``int``) number of columns in the 2-D matrix.
#' @param offset (``int``) diagonal offset from the main diagonal.        Default: if not provided, 0.
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of returned tensor.        Default: if ``None``, ``torch.long``.
#' @param device (:class:`torch.device`, optional) the desired device of returned tensor.        Default: if ``None``, uses the current device for the default tensor type        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU        for CPU tensor types and the current CUDA device for CUDA tensor types.
#' @param layout (:class:`torch.layout`, optional) currently only support ``torch.strided``.
#'
#' @examples
#' \dontrun{
#' a = torch_tril_indices(3, 3)
#' a
#' a = torch_tril_indices(4, 3, -1)
#' a
#' a = torch_tril_indices(4, 3, 1)
#' a
#' }
#'
#' @name torch_tril_indices
#'
#' @export
NULL


#' Triu_indices
#'
#' Returns the indices of the upper triangular part of a `row` by
#' `col` matrix in a 2-by-N Tensor, where the first row contains row
#' coordinates of all indices and the second row contains column coordinates.
#' Indices are ordered based on rows and then columns.
#' 
#' The upper triangular part of the matrix is defined as the elements on and
#' above the diagonal.
#' 
#' The argument `offset` controls which diagonal to consider. If
#' `offset` = 0, all elements on and above the main diagonal are
#' retained. A positive value excludes just as many diagonals above the main
#' diagonal, and similarly a negative value includes just as many diagonals below
#' the main diagonal. The main diagonal are the set of indices
#' \eqn{\lbrace (i, i) \rbrace} for \eqn{i \in [0, \min\{d_{1}, d_{2}\} - 1]}
#' where \eqn{d_{1}, d_{2}} are the dimensions of the matrix.
#' 
#' NOTE: when running on 'cuda', row * col must be less than \eqn{2^{59}} to
#' prevent overflow during calculation.
#'
#' @section Signatures:
#' 
#' triu_indices(row, col, offset=0, dtype=torch.long, device='cpu', layout=torch.strided) -> Tensor
#'
#'
#' @param row (``int``) number of rows in the 2-D matrix.
#' @param col (``int``) number of columns in the 2-D matrix.
#' @param offset (``int``) diagonal offset from the main diagonal.        Default: if not provided, 0.
#' @param dtype (:class:`torch.dtype`, optional) the desired data type of returned tensor.        Default: if ``None``, ``torch.long``.
#' @param device (:class:`torch.device`, optional) the desired device of returned tensor.        Default: if ``None``, uses the current device for the default tensor type        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU        for CPU tensor types and the current CUDA device for CUDA tensor types.
#' @param layout (:class:`torch.layout`, optional) currently only support ``torch.strided``.
#'
#' @examples
#' \dontrun{
#' a = torch_triu_indices(3, 3)
#' a
#' a = torch_triu_indices(4, 3, -1)
#' a
#' a = torch_triu_indices(4, 3, 1)
#' a
#' }
#'
#' @name torch_triu_indices
#'
#' @export
NULL


#' Trace
#'
#' Returns the sum of the elements of the diagonal of the input 2-D matrix.
#'
#' @section Signatures:
#' 
#' trace(input) -> Tensor
#'
#'
#'
#'
#' @examples
#' \dontrun{
#' x = torch_arange(1., 10.).view(3, 3)
#' x
#' torch_trace(x)
#' }
#'
#' @name torch_trace
#'
#' @export
NULL


#' Ne
#'
#' Computes \eqn{input \neq other} element-wise.
#' 
#' The second argument can be a number or a tensor whose shape is
#' :ref:`broadcastable <broadcasting-semantics>` with the first argument.
#'
#' @section Signatures:
#' 
#' ne(input, other, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the tensor to compare
#' @param other (Tensor or float) the tensor or value to compare
#' @param out (Tensor, optional) the output tensor that must be a `BoolTensor`
#'
#' @examples
#' \dontrun{
#' torch_ne(torch_tensor([[1, 2], [3, 4]]), torch_tensor([[1, 1], [4, 4]]))
#' }
#'
#' @name torch_ne
#'
#' @export
NULL


#' Eq
#'
#' Computes element-wise equality
#' 
#' The second argument can be a number or a tensor whose shape is
#' :ref:`broadcastable <broadcasting-semantics>` with the first argument.
#'
#' @section Signatures:
#' 
#' eq(input, other, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the tensor to compare
#' @param other (Tensor or float) the tensor or value to compare
#' @param out (Tensor, optional) the output tensor. Must be a `ByteTensor`
#'
#' @examples
#' \dontrun{
#' torch_eq(torch_tensor([[1, 2], [3, 4]]), torch_tensor([[1, 1], [4, 4]]))
#' }
#'
#' @name torch_eq
#'
#' @export
NULL


#' Ge
#'
#' Computes \eqn{\text{input} \geq \text{other}} element-wise.
#' 
#' The second argument can be a number or a tensor whose shape is
#' :ref:`broadcastable <broadcasting-semantics>` with the first argument.
#'
#' @section Signatures:
#' 
#' ge(input, other, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the tensor to compare
#' @param other (Tensor or float) the tensor or value to compare
#' @param out (Tensor, optional) the output tensor that must be a `BoolTensor`
#'
#' @examples
#' \dontrun{
#' torch_ge(torch_tensor([[1, 2], [3, 4]]), torch_tensor([[1, 1], [4, 4]]))
#' }
#'
#' @name torch_ge
#'
#' @export
NULL


#' Le
#'
#' Computes \eqn{\text{input} \leq \text{other}} element-wise.
#' 
#' The second argument can be a number or a tensor whose shape is
#' :ref:`broadcastable <broadcasting-semantics>` with the first argument.
#'
#' @section Signatures:
#' 
#' le(input, other, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the tensor to compare
#' @param other (Tensor or float) the tensor or value to compare
#' @param out (Tensor, optional) the output tensor that must be a `BoolTensor`
#'
#' @examples
#' \dontrun{
#' torch_le(torch_tensor([[1, 2], [3, 4]]), torch_tensor([[1, 1], [4, 4]]))
#' }
#'
#' @name torch_le
#'
#' @export
NULL


#' Gt
#'
#' Computes \eqn{\text{input} > \text{other}} element-wise.
#' 
#' The second argument can be a number or a tensor whose shape is
#' :ref:`broadcastable <broadcasting-semantics>` with the first argument.
#'
#' @section Signatures:
#' 
#' gt(input, other, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the tensor to compare
#' @param other (Tensor or float) the tensor or value to compare
#' @param out (Tensor, optional) the output tensor that must be a `BoolTensor`
#'
#' @examples
#' \dontrun{
#' torch_gt(torch_tensor([[1, 2], [3, 4]]), torch_tensor([[1, 1], [4, 4]]))
#' }
#'
#' @name torch_gt
#'
#' @export
NULL


#' Lt
#'
#' Computes \eqn{\text{input} < \text{other}} element-wise.
#' 
#' The second argument can be a number or a tensor whose shape is
#' :ref:`broadcastable <broadcasting-semantics>` with the first argument.
#'
#' @section Signatures:
#' 
#' lt(input, other, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the tensor to compare
#' @param other (Tensor or float) the tensor or value to compare
#' @param out (Tensor, optional) the output tensor that must be a `BoolTensor`
#'
#' @examples
#' \dontrun{
#' torch_lt(torch_tensor([[1, 2], [3, 4]]), torch_tensor([[1, 1], [4, 4]]))
#' }
#'
#' @name torch_lt
#'
#' @export
NULL


#' Take
#'
#' Returns a new tensor with the elements of `input` at the given indices.
#' The input tensor is treated as if it were viewed as a 1-D tensor. The result
#' takes the same shape as the indices.
#'
#' @section Signatures:
#' 
#' take(input, index) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param indices (LongTensor) the indices into tensor
#'
#' @examples
#' \dontrun{
#' src = torch_tensor([[4, 3, 5],
#' torch_take(src, torch_tensor([0, 2, 5]))
#' }
#'
#' @name torch_take
#'
#' @export
NULL


#' Index_select
#'
#' Returns a new tensor which indexes the `input` tensor along dimension
#' `dim` using the entries in `index` which is a `LongTensor`.
#' 
#' The returned tensor has the same number of dimensions as the original tensor
#' (`input`).  The `dim`\ th dimension has the same size as the length
#' of `index`; other dimensions have the same size as in the original tensor.
#' 
#' @note The returned tensor does **not** use the same storage as the original
#'           tensor.  If `out` has a different shape than expected, we
#'           silently change it to the correct shape, reallocating the underlying
#'           storage if necessary.
#'
#' @section Signatures:
#' 
#' index_select(input, dim, index, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param dim (int) the dimension in which we index
#' @param index (LongTensor) the 1-D tensor containing the indices to index
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' x = torch_randn(3, 4)
#' x
#' indices = torch_tensor([0, 2])
#' torch_index_select(x, 0, indices)
#' torch_index_select(x, 1, indices)
#' }
#'
#' @name torch_index_select
#'
#' @export
NULL


#' Masked_select
#'
#' Returns a new 1-D tensor which indexes the `input` tensor according to
#' the boolean mask `mask` which is a `BoolTensor`.
#' 
#' The shapes of the `mask` tensor and the `input` tensor don't need
#' to match, but they must be :ref:`broadcastable <broadcasting-semantics>`.
#' 
#' @note The returned tensor does **not** use the same storage
#'           as the original tensor
#'
#' @section Signatures:
#' 
#' masked_select(input, mask, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param mask (ByteTensor) the tensor containing the binary mask to index with
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' x = torch_randn(3, 4)
#' x
#' mask = x.ge(0.5)
#' mask
#' torch_masked_select(x, mask)
#' }
#'
#' @name torch_masked_select
#'
#' @export
NULL


#' Nonzero
#'
#' @note
#'     [`torch_nonzero(..., as_tuple=False) <torch.nonzero>`] (default) returns a
#'     2-D tensor where each row is the index for a nonzero value.
#' 
#'     [`torch_nonzero(..., as_tuple=True) <torch.nonzero>`] returns a tuple of 1-D
#'     index tensors, allowing for advanced indexing, so ``x[x.nonzero(as_tuple=True)]``
#'     gives all nonzero values of tensor ``x``. Of the returned tuple, each index tensor
#'     contains nonzero indices for a certain dimension.
#' 
#'     See below for more details on the two behaviors.
#' 
#' 
#' **When** `as_tuple` **is ``False`` (default)**:
#' 
#' Returns a tensor containing the indices of all non-zero elements of
#' `input`.  Each row in the result contains the indices of a non-zero
#' element in `input`. The result is sorted lexicographically, with
#' the last index changing the fastest (C-style).
#' 
#' If `input` has \eqn{n} dimensions, then the resulting indices tensor
#' `out` is of size \eqn{(z \times n)}, where \eqn{z} is the total number of
#' non-zero elements in the `input` tensor.
#' 
#' **When** `as_tuple` **is ``True``**:
#' 
#' Returns a tuple of 1-D tensors, one for each dimension in `input`,
#' each containing the indices (in that dimension) of all non-zero elements of
#' `input` .
#' 
#' If `input` has \eqn{n} dimensions, then the resulting tuple contains \eqn{n}
#' tensors of size \eqn{z}, where \eqn{z} is the total number of
#' non-zero elements in the `input` tensor.
#' 
#' As a special case, when `input` has zero dimensions and a nonzero scalar
#' value, it is treated as a one-dimensional tensor with one element.
#'
#' @section Signatures:
#' 
#' nonzero(input, *, out=None, as_tuple=False) -> LongTensor or tuple of LongTensors
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (LongTensor, optional) the output tensor containing indices
#'
#' @examples
#' \dontrun{
#' torch_nonzero(torch_tensor([1, 1, 1, 0, 1]))
#' torch_nonzero(torch_tensor([[0.6, 0.0, 0.0, 0.0],
#' torch_nonzero(torch_tensor([1, 1, 1, 0, 1]), as_tuple=True)
#' torch_nonzero(torch_tensor([[0.6, 0.0, 0.0, 0.0],
#' torch_nonzero(torch_tensor(5), as_tuple=True)
#' }
#'
#' @name torch_nonzero
#'
#' @export
NULL


#' Gather
#'
#' Gathers values along an axis specified by `dim`.
#' 
#' For a 3-D tensor the output is specified by::
#' 
#'     out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
#'     out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
#'     out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
#' 
#' If `input` is an n-dimensional tensor with size
#' \eqn{(x_0, x_1..., x_{i-1}, x_i, x_{i+1}, ..., x_{n-1})}
#' and ``dim = i``, then `index` must be an \eqn{n}-dimensional tensor with
#' size \eqn{(x_0, x_1, ..., x_{i-1}, y, x_{i+1}, ..., x_{n-1})} where \eqn{y \geq 1}
#' and `out` will have the same size as `index`.
#'
#' @section Signatures:
#' 
#' gather(input, dim, index, out=None, sparse_grad=False) -> Tensor
#'
#'
#' @param input (Tensor) the source tensor
#' @param dim (int) the axis along which to index
#' @param index (LongTensor) the indices of elements to gather
#' @param out (Tensor, optional) the destination tensor
#' @param sparse_grad (bool,optional) If ``True``, gradient w.r.t. :attr:`input` will be a sparse tensor.
#'
#' @examples
#' \dontrun{
#' t = torch_tensor([[1,2],[3,4]])
#' torch_gather(t, 1, torch_tensor([[0,0],[1,0]]))
#' }
#'
#' @name torch_gather
#'
#' @export
NULL


#' Addcmul
#'
#' Performs the element-wise multiplication of `tensor1`
#' by `tensor2`, multiply the result by the scalar `value`
#' and add it to `input`.
#' 
#' \deqn{
#'     \text{out}_i = \text{input}_i + \text{value} \times \text{tensor1}_i \times \text{tensor2}_i
#' }
#' The shapes of `tensor`, `tensor1`, and `tensor2` must be
#' :ref:`broadcastable <broadcasting-semantics>`.
#' 
#' For inputs of type `FloatTensor` or `DoubleTensor`, `value` must be
#' a real number, otherwise an integer.
#'
#' @section Signatures:
#' 
#' addcmul(input, value=1, tensor1, tensor2, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the tensor to be added
#' @param value (Number, optional) multiplier for \eqn{tensor1 .* tensor2}
#' @param tensor1 (Tensor) the tensor to be multiplied
#' @param tensor2 (Tensor) the tensor to be multiplied
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' t = torch_randn(1, 3)
#' t1 = torch_randn(3, 1)
#' t2 = torch_randn(1, 3)
#' torch_addcmul(t, 0.1, t1, t2)
#' }
#'
#' @name torch_addcmul
#'
#' @export
NULL


#' Addcdiv
#'
#' Performs the element-wise division of `tensor1` by `tensor2`,
#' multiply the result by the scalar `value` and add it to `input`.
#' 
#' \deqn{
#'     \text{out}_i = \text{input}_i + \text{value} \times \frac{\text{tensor1}_i}{\text{tensor2}_i}
#' }
#' The shapes of `input`, `tensor1`, and `tensor2` must be
#' :ref:`broadcastable <broadcasting-semantics>`.
#' 
#' For inputs of type `FloatTensor` or `DoubleTensor`, `value` must be
#' a real number, otherwise an integer.
#'
#' @section Signatures:
#' 
#' addcdiv(input, value=1, tensor1, tensor2, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the tensor to be added
#' @param value (Number, optional) multiplier for \eqn{\text{tensor1} / \text{tensor2}}
#' @param tensor1 (Tensor) the numerator tensor
#' @param tensor2 (Tensor) the denominator tensor
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' t = torch_randn(1, 3)
#' t1 = torch_randn(3, 1)
#' t2 = torch_randn(1, 3)
#' torch_addcdiv(t, 0.1, t1, t2)
#' }
#'
#' @name torch_addcdiv
#'
#' @export
NULL


#' Lstsq
#'
#' Computes the solution to the least squares and least norm problems for a full
#' rank matrix \eqn{A} of size \eqn{(m \times n)} and a matrix \eqn{B} of
#' size \eqn{(m \times k)}.
#' 
#' If \eqn{m \geq n}, [`lstsq`] solves the least-squares problem:
#' 
#' \deqn{
#' 
#'    \begin{array}{ll}
#'    \min_X & \|AX-B\|_2.
#'    \end{array}
#' }
#' If \eqn{m < n}, [`lstsq`] solves the least-norm problem:
#' 
#' \deqn{
#' 
#'    \begin{array}{ll}
#'    \min_X & \|X\|_2 & \text{subject to} & AX = B.
#'    \end{array}
#' }
#' Returned tensor \eqn{X} has shape \eqn{(\max(m, n) \times k)}. The first \eqn{n}
#' rows of \eqn{X} contains the solution. If \eqn{m \geq n}, the residual sum of squares
#' for the solution in each column is given by the sum of squares of elements in the
#' remaining \eqn{m - n} rows of that column.
#' 
#' @note
#'     The case when \eqn{m < n} is not supported on the GPU.
#'
#' @section Signatures:
#' 
#' lstsq(input, A, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the matrix \eqn{B}
#' @param A (Tensor) the \eqn{m} by \eqn{n} matrix \eqn{A}
#' @param out (tuple, optional) the optional destination tensor
#'
#' @examples
#' \dontrun{
#' A = torch_tensor([[1., 1, 1],
#' B = torch_tensor([[-10., -3],
#' X, _ = torch_lstsq(B, A)
#' X
#' }
#'
#' @name torch_lstsq
#'
#' @export
NULL


#' Triangular_solve
#'
#' Solves a system of equations with a triangular coefficient matrix \eqn{A}
#' and multiple right-hand sides \eqn{b}.
#' 
#' In particular, solves \eqn{AX = b} and assumes \eqn{A} is upper-triangular
#' with the default keyword arguments.
#' 
#' `torch_triangular_solve(b, A)` can take in 2D inputs `b, A` or inputs that are
#' batches of 2D matrices. If the inputs are batches, then returns
#' batched outputs `X`
#'
#' @section Signatures:
#' 
#' triangular_solve(input, A, upper=True, transpose=False, unitriangular=False) -> (Tensor, Tensor)
#'
#'
#' @param input (Tensor) multiple right-hand sides of size \eqn{(*, m, k)} where                \eqn{*} is zero of more batch dimensions (\eqn{b})
#' @param A (Tensor) the input triangular coefficient matrix of size \eqn{(*, m, m)}                where \eqn{*} is zero or more batch dimensions
#' @param upper (bool, optional) whether to solve the upper-triangular system        of equations (default) or the lower-triangular system of equations. Default: ``True``.
#' @param transpose (bool, optional) whether \eqn{A} should be transposed before        being sent into the solver. Default: ``False``.
#' @param unitriangular (bool, optional) whether \eqn{A} is unit triangular.        If True, the diagonal elements of \eqn{A} are assumed to be        1 and not referenced from \eqn{A}. Default: ``False``.
#'
#' 
#'
#' @name torch_triangular_solve
#'
#' @export
NULL


#' Symeig
#'
#' This function returns eigenvalues and eigenvectors
#' of a real symmetric matrix `input` or a batch of real symmetric matrices,
#' represented by a namedtuple (eigenvalues, eigenvectors).
#' 
#' This function calculates all eigenvalues (and vectors) of `input`
#' such that \eqn{\text{input} = V \text{diag}(e) V^T}.
#' 
#' The boolean argument `eigenvectors` defines computation of
#' both eigenvectors and eigenvalues or eigenvalues only.
#' 
#' If it is ``False``, only eigenvalues are computed. If it is ``True``,
#' both eigenvalues and eigenvectors are computed.
#' 
#' Since the input matrix `input` is supposed to be symmetric,
#' only the upper triangular portion is used by default.
#' 
#' If `upper` is ``False``, then lower triangular portion is used.
#' 
#' @note The eigenvalues are returned in ascending order. If `input` is a batch of matrices,
#'           then the eigenvalues of each matrix in the batch is returned in ascending order.
#' 
#' @note Irrespective of the original strides, the returned matrix `V` will
#'           be transposed, i.e. with strides `V.contiguous().transpose(-1, -2).stride()`.
#' 
#' @note Extra care needs to be taken when backward through outputs. Such
#'           operation is really only stable when all eigenvalues are distinct.
#'           Otherwise, ``NaN`` can appear as the gradients are not properly defined.
#'
#' @section Signatures:
#' 
#' symeig(input, eigenvectors=False, upper=True, out=None) -> (Tensor, Tensor)
#'
#'
#' @param input (Tensor) the input tensor of size \eqn{(*, n, n)} where `*` is zero or more                    batch dimensions consisting of symmetric matrices.
#' @param eigenvectors (boolean, optional) controls whether eigenvectors have to be computed
#' @param upper (boolean, optional) controls whether to consider upper-triangular or lower-triangular region
#' @param out (tuple, optional) the output tuple of (Tensor, Tensor)
#'
#' 
#'
#' @name torch_symeig
#'
#' @export
NULL


#' Eig
#'
#' Computes the eigenvalues and eigenvectors of a real square matrix.
#' 
#' @note
#'     Since eigenvalues and eigenvectors might be complex, backward pass is supported only
#'     for [`torch_symeig`]
#'
#' @section Signatures:
#' 
#' eig(input, eigenvectors=False, out=None) -> (Tensor, Tensor)
#'
#'
#' @param input (Tensor) the square matrix of shape \eqn{(n \times n)} for which the eigenvalues and eigenvectors        will be computed
#' @param eigenvectors (bool) ``True`` to compute both eigenvalues and eigenvectors;        otherwise, only eigenvalues will be computed
#' @param out (tuple, optional) the output tensors
#'
#' 
#'
#' @name torch_eig
#'
#' @export
NULL


#' Svd
#'
#' This function returns a namedtuple ``(U, S, V)`` which is the singular value
#' decomposition of a input real matrix or batches of real matrices `input` such that
#' \eqn{input = U \times diag(S) \times V^T}.
#' 
#' If `some` is ``True`` (default), the method returns the reduced singular value decomposition
#' i.e., if the last two dimensions of `input` are ``m`` and ``n``, then the returned
#' `U` and `V` matrices will contain only \eqn{min(n, m)} orthonormal columns.
#' 
#' If `compute_uv` is ``False``, the returned `U` and `V` matrices will be zero matrices
#' of shape \eqn{(m \times m)} and \eqn{(n \times n)} respectively. `some` will be ignored here.
#' 
#' @note The singular values are returned in descending order. If `input` is a batch of matrices,
#'           then the singular values of each matrix in the batch is returned in descending order.
#' 
#' @note The implementation of SVD on CPU uses the LAPACK routine `?gesdd` (a divide-and-conquer
#'           algorithm) instead of `?gesvd` for speed. Analogously, the SVD on GPU uses the MAGMA routine
#'           `gesdd` as well.
#' 
#' @note Irrespective of the original strides, the returned matrix `U`
#'           will be transposed, i.e. with strides :code:`U.contiguous().transpose(-2, -1).stride()`
#' 
#' @note Extra care needs to be taken when backward through `U` and `V`
#'           outputs. Such operation is really only stable when `input` is
#'           full rank with all distinct singular values. Otherwise, ``NaN`` can
#'           appear as the gradients are not properly defined. Also, notice that
#'           double backward will usually do an additional backward through `U` and
#'           `V` even if the original backward is only on `S`.
#' 
#' @note When `some` = ``False``, the gradients on :code:`U[..., :, min(m, n):]`
#'           and :code:`V[..., :, min(m, n):]` will be ignored in backward as those vectors
#'           can be arbitrary bases of the subspaces.
#' 
#' @note When `compute_uv` = ``False``, backward cannot be performed since `U` and `V`
#'           from the forward pass is required for the backward operation.
#'
#' @section Signatures:
#' 
#' svd(input, some=True, compute_uv=True, out=None) -> (Tensor, Tensor, Tensor)
#'
#'
#' @param input (Tensor) the input tensor of size \eqn{(*, m, n)} where `*` is zero or more                    batch dimensions consisting of \eqn{m \times n} matrices.
#' @param some (bool, optional) controls the shape of returned `U` and `V`
#' @param compute_uv (bool, optional) option whether to compute `U` and `V` or not
#' @param out (tuple, optional) the output tuple of tensors
#'
#' @examples
#' \dontrun{
#' a = torch_randn(5, 3)
#' a
#' u, s, v = torch_svd(a)
#' u
#' s
#' v
#' torch_dist(a, torch_mm(torch_mm(u, torch_diag(s)), v.t()))
#' a_big = torch_randn(7, 5, 3)
#' u, s, v = torch_svd(a_big)
#' torch_dist(a_big, torch_matmul(torch_matmul(u, torch_diag_embed(s)), v.transpose(-2, -1)))
#' }
#'
#' @name torch_svd
#'
#' @export
NULL


#' Cholesky
#'
#' Computes the Cholesky decomposition of a symmetric positive-definite
#' matrix \eqn{A} or for batches of symmetric positive-definite matrices.
#' 
#' If `upper` is ``True``, the returned matrix ``U`` is upper-triangular, and
#' the decomposition has the form:
#' 
#' \deqn{
#' 
#'   A = U^TU
#' }
#' If `upper` is ``False``, the returned matrix ``L`` is lower-triangular, and
#' the decomposition has the form:
#' 
#' \deqn{
#' 
#'     A = LL^T
#' }
#' If `upper` is ``True``, and \eqn{A} is a batch of symmetric positive-definite
#' matrices, then the returned tensor will be composed of upper-triangular Cholesky factors
#' of each of the individual matrices. Similarly, when `upper` is ``False``, the returned
#' tensor will be composed of lower-triangular Cholesky factors of each of the individual
#' matrices.
#'
#' @section Signatures:
#' 
#' cholesky(input, upper=False, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor \eqn{A} of size \eqn{(*, n, n)} where `*` is zero or more                batch dimensions consisting of symmetric positive-definite matrices.
#' @param upper (bool, optional) flag that indicates whether to return a                            upper or lower triangular matrix. Default: ``False``
#' @param out (Tensor, optional) the output matrix
#'
#' @examples
#' \dontrun{
#' a = torch_randn(3, 3)
#' a = torch_mm(a, a.t()) # make symmetric positive-definite
#' l = torch_cholesky(a)
#' a
#' l
#' torch_mm(l, l.t())
#' a = torch_randn(3, 2, 2)
#' a = torch_matmul(a, a.transpose(-1, -2)) + 1e-03 # make symmetric positive-definite
#' l = torch_cholesky(a)
#' z = torch_matmul(l, l.transpose(-1, -2))
#' torch_max(torch_abs(z - a)) # Max non-zero
#' }
#'
#' @name torch_cholesky
#'
#' @export
NULL


#' Cholesky_solve
#'
#' Solves a linear system of equations with a positive semidefinite
#' matrix to be inverted given its Cholesky factor matrix \eqn{u}.
#' 
#' If `upper` is ``False``, \eqn{u} is and lower triangular and `c` is
#' returned such that:
#' 
#' \deqn{
#'     c = (u u^T)^{{-1}} b
#' }
#' If `upper` is ``True`` or not provided, \eqn{u} is upper triangular
#' and `c` is returned such that:
#' 
#' \deqn{
#'     c = (u^T u)^{{-1}} b
#' }
#' `torch_cholesky_solve(b, u)` can take in 2D inputs `b, u` or inputs that are
#' batches of 2D matrices. If the inputs are batches, then returns
#' batched outputs `c`
#'
#' @section Signatures:
#' 
#' cholesky_solve(input, input2, upper=False, out=None) -> Tensor
#'
#'
#' @param input (Tensor) input matrix \eqn{b} of size \eqn{(*, m, k)},                where \eqn{*} is zero or more batch dimensions
#' @param input2 (Tensor) input matrix \eqn{u} of size \eqn{(*, m, m)},                where \eqn{*} is zero of more batch dimensions composed of                upper or lower triangular Cholesky factor
#' @param upper (bool, optional) whether to consider the Cholesky factor as a                            lower or upper triangular matrix. Default: ``False``.
#' @param out (Tensor, optional) the output tensor for `c`
#'
#' @examples
#' \dontrun{
#' a = torch_randn(3, 3)
#' a = torch_mm(a, a.t()) # make symmetric positive definite
#' u = torch_cholesky(a)
#' a
#' b = torch_randn(3, 2)
#' b
#' torch_cholesky_solve(b, u)
#' torch_mm(a.inverse(), b)
#' }
#'
#' @name torch_cholesky_solve
#'
#' @export
NULL


#' Solve
#'
#' This function returns the solution to the system of linear
#' equations represented by \eqn{AX = B} and the LU factorization of
#' A, in order as a namedtuple `solution, LU`.
#' 
#' `LU` contains `L` and `U` factors for LU factorization of `A`.
#' 
#' `torch_solve(B, A)` can take in 2D inputs `B, A` or inputs that are
#' batches of 2D matrices. If the inputs are batches, then returns
#' batched outputs `solution, LU`.
#' 
#' @note
#' 
#'     Irrespective of the original strides, the returned matrices
#'     `solution` and `LU` will be transposed, i.e. with strides like
#'     `B.contiguous().transpose(-1, -2).stride()` and
#'     `A.contiguous().transpose(-1, -2).stride()` respectively.
#'
#' @section Signatures:
#' 
#' torch.solve(input, A, out=None) -> (Tensor, Tensor)
#'
#'
#' @param input (Tensor) input matrix \eqn{B} of size \eqn{(*, m, k)} , where \eqn{*}                is zero or more batch dimensions.
#' @param A (Tensor) input square matrix of size \eqn{(*, m, m)}, where                \eqn{*} is zero or more batch dimensions.
#' @param out ((Tensor, Tensor) optional output tuple.
#'
#' @examples
#' \dontrun{
#' A = torch_tensor([[6.80, -2.11,  5.66,  5.97,  8.23],
#' B = torch_tensor([[4.02,  6.19, -8.22, -7.57, -3.03],
#' X, LU = torch_solve(B, A)
#' torch_dist(B, torch_mm(A, X))
#' # Batched solver example
#' A = torch_randn(2, 3, 1, 4, 4)
#' B = torch_randn(2, 3, 1, 4, 6)
#' X, LU = torch_solve(B, A)
#' torch_dist(B, A.matmul(X))
#' }
#'
#' @name torch_solve
#'
#' @export
NULL


#' Cholesky_inverse
#'
#' Computes the inverse of a symmetric positive-definite matrix \eqn{A} using its
#' Cholesky factor \eqn{u}: returns matrix ``inv``. The inverse is computed using
#' LAPACK routines ``dpotri`` and ``spotri`` (and the corresponding MAGMA routines).
#' 
#' If `upper` is ``False``, \eqn{u} is lower triangular
#' such that the returned tensor is
#' 
#' \deqn{
#'     inv = (uu^{{T}})^{{-1}}
#' }
#' If `upper` is ``True`` or not provided, \eqn{u} is upper
#' triangular such that the returned tensor is
#' 
#' \deqn{
#'     inv = (u^T u)^{{-1}}
#' }
#'
#' @section Signatures:
#' 
#' cholesky_inverse(input, upper=False, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input 2-D tensor \eqn{u}, a upper or lower triangular           Cholesky factor
#' @param upper (bool, optional) whether to return a lower (default) or upper triangular matrix
#' @param out (Tensor, optional) the output tensor for `inv`
#'
#' @examples
#' \dontrun{
#' a = torch_randn(3, 3)
#' a = torch_mm(a, a.t()) + 1e-05 * torch_eye(3) # make symmetric positive definite
#' u = torch_cholesky(a)
#' a
#' torch_cholesky_inverse(u)
#' a.inverse()
#' }
#'
#' @name torch_cholesky_inverse
#'
#' @export
NULL


#' Qr
#'
#' Computes the QR decomposition of a matrix or a batch of matrices `input`,
#' and returns a namedtuple (Q, R) of tensors such that \eqn{\text{input} = Q R}
#' with \eqn{Q} being an orthogonal matrix or batch of orthogonal matrices and
#' \eqn{R} being an upper triangular matrix or batch of upper triangular matrices.
#' 
#' If `some` is ``True``, then this function returns the thin (reduced) QR factorization.
#' Otherwise, if `some` is ``False``, this function returns the complete QR factorization.
#' 
#' @note precision may be lost if the magnitudes of the elements of `input`
#'           are large
#' 
#' @note While it should always give you a valid decomposition, it may not
#'           give you the same one across platforms - it will depend on your
#'           LAPACK implementation.
#'
#' @section Signatures:
#' 
#' qr(input, some=True, out=None) -> (Tensor, Tensor)
#'
#'
#' @param input (Tensor) the input tensor of size \eqn{(*, m, n)} where `*` is zero or more                batch dimensions consisting of matrices of dimension \eqn{m \times n}.
#' @param some (bool, optional) Set to ``True`` for reduced QR decomposition and ``False`` for                complete QR decomposition.
#' @param out (tuple, optional) tuple of `Q` and `R` tensors                satisfying :code:`input = torch.matmul(Q, R)`.                The dimensions of `Q` and `R` are \eqn{(*, m, k)} and \eqn{(*, k, n)}                respectively, where \eqn{k = \min(m, n)} if :attr:`some:` is ``True`` and                \eqn{k = m} otherwise.
#'
#' @examples
#' \dontrun{
#' a = torch_tensor([[12., -51, 4], [6, 167, -68], [-4, 24, -41]])
#' q, r = torch_qr(a)
#' q
#' r
#' torch_mm(q, r).round()
#' torch_mm(q.t(), q).round()
#' a = torch_randn(3, 4, 5)
#' q, r = torch_qr(a, some=False)
#' torch_allclose(torch_matmul(q, r), a)
#' torch_allclose(torch_matmul(q.transpose(-2, -1), q), torch_eye(5))
#' }
#'
#' @name torch_qr
#'
#' @export
NULL


#' Geqrf
#'
#' This is a low-level function for calling LAPACK directly. This function
#' returns a namedtuple (a, tau) as defined in `LAPACK documentation for geqrf`_ .
#' 
#' You'll generally want to use [`torch_qr`] instead.
#' 
#' Computes a QR decomposition of `input`, but without constructing
#' \eqn{Q} and \eqn{R} as explicit separate matrices.
#' 
#' Rather, this directly calls the underlying LAPACK function `?geqrf`
#' which produces a sequence of 'elementary reflectors'.
#' 
#' See `LAPACK documentation for geqrf`_ for further details.
#'
#' @section Signatures:
#' 
#' geqrf(input, out=None) -> (Tensor, Tensor)
#'
#'
#' @param input (Tensor) the input matrix
#' @param out (tuple, optional) the output tuple of (Tensor, Tensor)
#'
#' 
#'
#' @name torch_geqrf
#'
#' @export
NULL


#' Orgqr
#'
#' Computes the orthogonal matrix `Q` of a QR factorization, from the `(input, input2)`
#' tuple returned by [`torch_geqrf`].
#' 
#' This directly calls the underlying LAPACK function `?orgqr`.
#' See `LAPACK documentation for orgqr`_ for further details.
#'
#' @section Signatures:
#' 
#' orgqr(input, input2) -> Tensor
#'
#'
#' @param input (Tensor) the `a` from :func:`torch.geqrf`.
#' @param input2 (Tensor) the `tau` from :func:`torch.geqrf`.
#'
#' 
#'
#' @name torch_orgqr
#'
#' @export
NULL


#' Ormqr
#'
#' Multiplies `mat` (given by `input3`) by the orthogonal `Q` matrix of the QR factorization
#' formed by [`torch_geqrf`] that is represented by `(a, tau)` (given by (`input`, `input2`)).
#' 
#' This directly calls the underlying LAPACK function `?ormqr`.
#' See `LAPACK documentation for ormqr`_ for further details.
#'
#' @section Signatures:
#' 
#' ormqr(input, input2, input3, left=True, transpose=False) -> Tensor
#'
#'
#' @param input (Tensor) the `a` from :func:`torch.geqrf`.
#' @param input2 (Tensor) the `tau` from :func:`torch.geqrf`.
#' @param input3 (Tensor) the matrix to be multiplied.
#'
#' 
#'
#' @name torch_ormqr
#'
#' @export
NULL


#' Lu_solve
#'
#' Returns the LU solve of the linear system \eqn{Ax = b} using the partially pivoted
#' LU factorization of A from :meth:`torch_lu`.
#'
#' @section Signatures:
#' 
#' lu_solve(input, LU_data, LU_pivots, out=None) -> Tensor
#'
#'
#' @param b (Tensor) the RHS tensor of size \eqn{(*, m, k)}, where \eqn{*}                is zero or more batch dimensions.
#' @param LU_data (Tensor) the pivoted LU factorization of A from :meth:`torch.lu` of size \eqn{(*, m, m)},                       where \eqn{*} is zero or more batch dimensions.
#' @param LU_pivots (IntTensor) the pivots of the LU factorization from :meth:`torch.lu` of size \eqn{(*, m)},                           where \eqn{*} is zero or more batch dimensions.                           The batch dimensions of :attr:`LU_pivots` must be equal to the batch dimensions of                           :attr:`LU_data`.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' A = torch_randn(2, 3, 3)
#' b = torch_randn(2, 3, 1)
#' A_LU = torch_lu(A)
#' x = torch_lu_solve(b, *A_LU)
#' torch_norm(torch_bmm(A, x) - b)
#' }
#'
#' @name torch_lu_solve
#'
#' @export
NULL


#' Multinomial
#'
#' Returns a tensor where each row contains `num_samples` indices sampled
#' from the multinomial probability distribution located in the corresponding row
#' of tensor `input`.
#' 
#' @note
#'     The rows of `input` do not need to sum to one (in which case we use
#'     the values as weights), but must be non-negative, finite and have
#'     a non-zero sum.
#' 
#' Indices are ordered from left to right according to when each was sampled
#' (first samples are placed in first column).
#' 
#' If `input` is a vector, `out` is a vector of size `num_samples`.
#' 
#' If `input` is a matrix with `m` rows, `out` is an matrix of shape
#' \eqn{(m \times \text{num\_samples})}.
#' 
#' If replacement is ``True``, samples are drawn with replacement.
#' 
#' If not, they are drawn without replacement, which means that when a
#' sample index is drawn for a row, it cannot be drawn again for that row.
#' 
#' @note
#'     When drawn without replacement, `num_samples` must be lower than
#'     number of non-zero elements in `input` (or the min number of non-zero
#'     elements in each row of `input` if it is a matrix).
#'
#' @section Signatures:
#' 
#' multinomial(input, num_samples, replacement=False, *, generator=None, out=None) -> LongTensor
#'
#'
#' @param input (Tensor) the input tensor containing probabilities
#' @param num_samples (int) number of samples to draw
#' @param replacement (bool, optional) whether to draw with replacement or not
#' @param generator (:class:`torch.Generator`, optional) a pseudorandom number generator for sampling
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' weights = torch_tensor([0, 10, 3, 0], dtype=torch_float) # create a tensor of weights
#' torch_multinomial(weights, 2)
#' torch_multinomial(weights, 4) # ERROR!
#' torch_multinomial(weights, 4, replacement=True)
#' }
#'
#' @name torch_multinomial
#'
#' @export
NULL


#' Lgamma
#'
#' Computes the logarithm of the gamma function on `input`.
#' 
#' \deqn{
#'     \text{out}_{i} = \log \Gamma(\text{input}_{i})
#' }
#'
#' @section Signatures:
#' 
#' lgamma(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_arange(0.5, 2, 0.5)
#' torch_lgamma(a)
#' }
#'
#' @name torch_lgamma
#'
#' @export
NULL


#' Digamma
#'
#' Computes the logarithmic derivative of the gamma function on `input`.
#' 
#' \deqn{
#'     \psi(x) = \frac{d}{dx} \ln\left(\Gamma\left(x\right)\right) = \frac{\Gamma'(x)}{\Gamma(x)}
#' }
#'
#' @section Signatures:
#' 
#' digamma(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the tensor to compute the digamma function on
#'
#' @examples
#' \dontrun{
#' a = torch_tensor([1, 0.5])
#' torch_digamma(a)
#' }
#'
#' @name torch_digamma
#'
#' @export
NULL


#' Polygamma
#'
#' Computes the \eqn{n^{th}} derivative of the digamma function on `input`.
#' \eqn{n \geq 0} is called the order of the polygamma function.
#' 
#' \deqn{
#'     \psi^{(n)}(x) = \frac{d^{(n)}}{dx^{(n)}} \psi(x)
#' }
#' @note
#'     This function is not implemented for \eqn{n \geq 2}.
#'
#' @section Signatures:
#' 
#' polygamma(n, input, out=None) -> Tensor
#'
#'
#' @param n (int) the order of the polygamma function
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_tensor([1, 0.5])
#' torch_polygamma(1, a)
#' }
#'
#' @name torch_polygamma
#'
#' @export
NULL


#' Erfinv
#'
#' Computes the inverse error function of each element of `input`.
#' The inverse error function is defined in the range \eqn{(-1, 1)} as:
#' 
#' \deqn{
#'     \mathrm{erfinv}(\mathrm{erf}(x)) = x
#' }
#'
#' @section Signatures:
#' 
#' erfinv(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' torch_erfinv(torch_tensor([0, 0.5, -1.]))
#' }
#'
#' @name torch_erfinv
#'
#' @export
NULL


#' Sign
#'
#' Returns a new tensor with the signs of the elements of `input`.
#' 
#' \deqn{
#'     \text{out}_{i} = \operatorname{sgn}(\text{input}_{i})
#' }
#'
#' @section Signatures:
#' 
#' sign(input, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_tensor([0.7, -1.2, 0., 2.3])
#' a
#' torch_sign(a)
#' }
#'
#' @name torch_sign
#'
#' @export
NULL


#' Dist
#'
#' Returns the p-norm of (`input` - `other`)
#' 
#' The shapes of `input` and `other` must be
#' :ref:`broadcastable <broadcasting-semantics>`.
#'
#' @section Signatures:
#' 
#' dist(input, other, p=2) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param other (Tensor) the Right-hand-side input tensor
#' @param p (float, optional) the norm to be computed
#'
#' @examples
#' \dontrun{
#' x = torch_randn(4)
#' x
#' y = torch_randn(4)
#' y
#' torch_dist(x, y, 3.5)
#' torch_dist(x, y, 3)
#' torch_dist(x, y, 0)
#' torch_dist(x, y, 1)
#' }
#'
#' @name torch_dist
#'
#' @export
NULL


#' Atan2
#'
#' Element-wise arctangent of \eqn{\text{input}_{i} / \text{other}_{i}}
#' with consideration of the quadrant. Returns a new tensor with the signed angles
#' in radians between vector \eqn{(\text{other}_{i}, \text{input}_{i})}
#' and vector \eqn{(1, 0)}. (Note that \eqn{\text{other}_{i}}, the second
#' parameter, is the x-coordinate, while \eqn{\text{input}_{i}}, the first
#' parameter, is the y-coordinate.)
#' 
#' The shapes of ``input`` and ``other`` must be
#' :ref:`broadcastable <broadcasting-semantics>`.
#'
#' @section Signatures:
#' 
#' atan2(input, other, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the first input tensor
#' @param other (Tensor) the second input tensor
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4)
#' a
#' torch_atan2(a, torch_randn(4))
#' }
#'
#' @name torch_atan2
#'
#' @export
NULL


#' Lerp
#'
#' lerp(input, end, weight, out=None)
#' 
#' Does a linear interpolation of two tensors `start` (given by `input`) and `end` based
#' on a scalar or tensor `weight` and returns the resulting `out` tensor.
#' 
#' \deqn{
#'     \text{out}_i = \text{start}_i + \text{weight}_i \times (\text{end}_i - \text{start}_i)
#' }
#' The shapes of `start` and `end` must be
#' :ref:`broadcastable <broadcasting-semantics>`. If `weight` is a tensor, then
#' the shapes of `weight`, `start`, and `end` must be :ref:`broadcastable <broadcasting-semantics>`.
#'
#' 
#'
#' @param input (Tensor) the tensor with the starting points
#' @param end (Tensor) the tensor with the ending points
#' @param weight (float or tensor) the weight for the interpolation formula
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' start = torch_arange(1., 5.)
#' end = torch_empty(4).fill_(10)
#' start
#' end
#' torch_lerp(start, end, 0.5)
#' torch_lerp(start, end, torch_full_like(start, 0.5))
#' }
#'
#' @name torch_lerp
#'
#' @export
NULL


#' Histc
#'
#' Computes the histogram of a tensor.
#' 
#' The elements are sorted into equal width bins between `min` and
#' `max`. If `min` and `max` are both zero, the minimum and
#' maximum values of the data are used.
#'
#' @section Signatures:
#' 
#' histc(input, bins=100, min=0, max=0, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param bins (int) number of histogram bins
#' @param min (int) lower end of the range (inclusive)
#' @param max (int) upper end of the range (inclusive)
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' torch_histc(torch_tensor([1., 2, 1]), bins=4, min=0, max=3)
#' }
#'
#' @name torch_histc
#'
#' @export
NULL


#' Fmod
#'
#' Computes the element-wise remainder of division.
#' 
#' The dividend and divisor may contain both for integer and floating point
#' numbers. The remainder has the same sign as the dividend `input`.
#' 
#' When `other` is a tensor, the shapes of `input` and
#' `other` must be :ref:`broadcastable <broadcasting-semantics>`.
#'
#' @section Signatures:
#' 
#' fmod(input, other, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the dividend
#' @param other (Tensor or float) the divisor, which may be either a number or a tensor of the same shape as the dividend
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' torch_fmod(torch_tensor([-3., -2, -1, 1, 2, 3]), 2)
#' torch_fmod(torch_tensor([1., 2, 3, 4, 5]), 1.5)
#' }
#'
#' @name torch_fmod
#'
#' @export
NULL


#' Remainder
#'
#' Computes the element-wise remainder of division.
#' 
#' The divisor and dividend may contain both for integer and floating point
#' numbers. The remainder has the same sign as the divisor.
#' 
#' When `other` is a tensor, the shapes of `input` and
#' `other` must be :ref:`broadcastable <broadcasting-semantics>`.
#'
#' @section Signatures:
#' 
#' remainder(input, other, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the dividend
#' @param other (Tensor or float) the divisor that may be either a number or a                               Tensor of the same shape as the dividend
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' torch_remainder(torch_tensor([-3., -2, -1, 1, 2, 3]), 2)
#' torch_remainder(torch_tensor([1., 2, 3, 4, 5]), 1.5)
#' }
#'
#' @name torch_remainder
#'
#' @export
NULL


#' Sort
#'
#' Sorts the elements of the `input` tensor along a given dimension
#' in ascending order by value.
#' 
#' If `dim` is not given, the last dimension of the `input` is chosen.
#' 
#' If `descending` is ``True`` then the elements are sorted in descending
#' order by value.
#' 
#' A namedtuple of (values, indices) is returned, where the `values` are the
#' sorted values and `indices` are the indices of the elements in the original
#' `input` tensor.
#'
#' @section Signatures:
#' 
#' sort(input, dim=-1, descending=False, out=None) -> (Tensor, LongTensor)
#'
#'
#' @param input (Tensor) the input tensor.
#' @param dim (int, optional) the dimension to sort along
#' @param descending (bool, optional) controls the sorting order (ascending or descending)
#' @param out (tuple, optional) the output tuple of (`Tensor`, `LongTensor`) that can        be optionally given to be used as output buffers
#'
#' @examples
#' \dontrun{
#' x = torch_randn(3, 4)
#' sorted, indices = torch_sort(x)
#' sorted
#' indices
#' sorted, indices = torch_sort(x, 0)
#' sorted
#' indices
#' }
#'
#' @name torch_sort
#'
#' @export
NULL


#' Argsort
#'
#' Returns the indices that sort a tensor along a given dimension in ascending
#' order by value.
#' 
#' This is the second value returned by :meth:`torch_sort`.  See its documentation
#' for the exact semantics of this method.
#'
#' @section Signatures:
#' 
#' argsort(input, dim=-1, descending=False, out=None) -> LongTensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param dim (int, optional) the dimension to sort along
#' @param descending (bool, optional) controls the sorting order (ascending or descending)
#'
#' @examples
#' \dontrun{
#' a = torch_randn(4, 4)
#' a
#' torch_argsort(a, dim=1)
#' }
#'
#' @name torch_argsort
#'
#' @export
NULL


#' Topk
#'
#' Returns the `k` largest elements of the given `input` tensor along
#' a given dimension.
#' 
#' If `dim` is not given, the last dimension of the `input` is chosen.
#' 
#' If `largest` is ``False`` then the `k` smallest elements are returned.
#' 
#' A namedtuple of `(values, indices)` is returned, where the `indices` are the indices
#' of the elements in the original `input` tensor.
#' 
#' The boolean option `sorted` if ``True``, will make sure that the returned
#' `k` elements are themselves sorted
#'
#' @section Signatures:
#' 
#' topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)
#'
#'
#' @param input (Tensor) the input tensor.
#' @param k (int) the k in "top-k"
#' @param dim (int, optional) the dimension to sort along
#' @param largest (bool, optional) controls whether to return largest or           smallest elements
#' @param sorted (bool, optional) controls whether to return the elements           in sorted order
#' @param out (tuple, optional) the output tuple of (Tensor, LongTensor) that can be        optionally given to be used as output buffers
#'
#' @examples
#' \dontrun{
#' x = torch_arange(1., 6.)
#' x
#' torch_topk(x, 3)
#' }
#'
#' @name torch_topk
#'
#' @export
NULL


#' Renorm
#'
#' Returns a tensor where each sub-tensor of `input` along dimension
#' `dim` is normalized such that the `p`-norm of the sub-tensor is lower
#' than the value `maxnorm`
#' 
#' @note If the norm of a row is lower than `maxnorm`, the row is unchanged
#'
#' @section Signatures:
#' 
#' renorm(input, p, dim, maxnorm, out=None) -> Tensor
#'
#'
#' @param input (Tensor) the input tensor.
#' @param p (float) the power for the norm computation
#' @param dim (int) the dimension to slice over to get the sub-tensors
#' @param maxnorm (float) the maximum norm to keep each sub-tensor under
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' x = torch_ones(3, 3)
#' x[1].fill_(2)
#' x[2].fill_(3)
#' x
#' torch_renorm(x, 1, 0, 5)
#' }
#'
#' @name torch_renorm
#'
#' @export
NULL


#' Equal
#'
#' ``True`` if two tensors have the same size and elements, ``False`` otherwise.
#'
#' @section Signatures:
#' 
#' equal(input, other) -> bool
#'
#'
#'
#'
#' @examples
#' \dontrun{
#' torch_equal(torch_tensor([1, 2]), torch_tensor([1, 2]))
#' }
#'
#' @name torch_equal
#'
#' @export
NULL


#' Normal
#'
#' Returns a tensor of random numbers drawn from separate normal distributions
#' whose mean and standard deviation are given.
#' 
#' The `mean` is a tensor with the mean of
#' each output element's normal distribution
#' 
#' The `std` is a tensor with the standard deviation of
#' each output element's normal distribution
#' 
#' The shapes of `mean` and `std` don't need to match, but the
#' total number of elements in each tensor need to be the same.
#' 
#' @note When the shapes do not match, the shape of `mean`
#'           is used as the shape for the returned output tensor
#'
#' @section Signatures:
#' 
#' normal(mean, std, *, generator=None, out=None) -> Tensor
#'
#'
#' @param mean (Tensor) the tensor of per-element means
#' @param std (Tensor) the tensor of per-element standard deviations
#' @param generator (:class:`torch.Generator`, optional) a pseudorandom number generator for sampling
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' torch_normal(mean=torch_arange(1., 11.), std=torch_arange(1, 0, -0.1))
#' }
#'
#' @name torch_normal
#'
#' @export
NULL


#' Normal
#'
#' Similar to the function above, but the means are shared among all drawn
#' elements.
#'
#' @section Signatures:
#' 
#' normal(mean=0.0, std, out=None) -> Tensor
#'
#'
#' @param mean (float, optional) the mean for all distributions
#' @param std (Tensor) the tensor of per-element standard deviations
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' torch_normal(mean=0.5, std=torch_arange(1., 6.))
#' }
#'
#' @name torch_normal
#'
#' @export
NULL


#' Normal
#'
#' Similar to the function above, but the standard-deviations are shared among
#' all drawn elements.
#'
#' @section Signatures:
#' 
#' normal(mean, std=1.0, out=None) -> Tensor
#'
#'
#' @param mean (Tensor) the tensor of per-element means
#' @param std (float, optional) the standard deviation for all distributions
#' @param out (Tensor, optional) the output tensor
#'
#' @examples
#' \dontrun{
#' torch_normal(mean=torch_arange(1., 6.))
#' }
#'
#' @name torch_normal
#'
#' @export
NULL


#' Normal
#'
#' Similar to the function above, but the means and standard deviations are shared
#' among all drawn elements. The resulting tensor has size given by `size`.
#'
#' @section Signatures:
#' 
#' normal(mean, std, size, *, out=None) -> Tensor
#'
#'
#' @param mean (float) the mean for all distributions
#' @param std (float) the standard deviation for all distributions
#' @param size (int...) a sequence of integers defining the shape of the output tensor.
#' @param out (Tensor, optional) the output tensor.
#'
#' @examples
#' \dontrun{
#' torch_normal(2, 3, size=(1, 4))
#' }
#'
#' @name torch_normal
#'
#' @export
NULL


#' Isfinite
#'
#' Returns a new tensor with boolean elements representing if each element is `Finite` or not.
#'
#' 
#'
#' @param tensor (Tensor) A tensor to check
#'
#' 
#'
#' @name torch_isfinite
#'
#' @export
NULL
