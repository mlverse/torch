#' Abs
#'
#' abs(input, out=None) -> Tensor
#' 
#' Computes the element-wise absolute value of the given :attr:`input` tensor.
#' 
#' .. math::
#'     \text{out}_{i} = |\text{input}_{i}|
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
#' angle(input, out=None) -> Tensor
#' 
#' Computes the element-wise angle (in radians) of the given :attr:`input` tensor.
#' 
#' .. math::
#'     \text{out}_{i} = angle(\text{input}_{i})
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
#' real(input, out=None) -> Tensor
#' 
#' Computes the element-wise real value of the given :attr:`input` tensor.
#' 
#' .. math::
#'     \text{out}_{i} = real(\text{input}_{i})
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
#' imag(input, out=None) -> Tensor
#' 
#' Computes the element-wise imag value of the given :attr:`input` tensor.
#' 
#' .. math::
#'     \text{out}_{i} = imag(\text{input}_{i})
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
#' conj(input, out=None) -> Tensor
#' 
#' Computes the element-wise conjugate of the given :attr:`input` tensor.
#' 
#' .. math::
#'     \text{out}_{i} = conj(\text{input}_{i})
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
#' acos(input, out=None) -> Tensor
#' 
#' Returns a new tensor with the arccosine  of the elements of :attr:`input`.
#' 
#' .. math::
#'     \text{out}_{i} = \cos^{-1}(\text{input}_{i})
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
#' avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True) -> Tensor
#' 
#' Applies a 1D average pooling over an input signal composed of several
#' input planes.
#' 
#' See :class:`~torch.nn.AvgPool1d` for details and output shape.
#'
#' @param input NA input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iW)`
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
#' adaptive_avg_pool1d(input, output_size) -> Tensor
#' 
#' Applies a 1D adaptive average pooling over an input signal composed of
#' several input planes.
#' 
#' See :class:`~torch.nn.AdaptiveAvgPool1d` for details and output shape.
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
#' Adds the scalar :attr:`other` to each element of the input :attr:`input`
#' and returns a new resulting tensor.
#' 
#' .. math::
#'     \text{out} = \text{input} + \text{other}
#' 
#' If :attr:`input` is of type FloatTensor or DoubleTensor, :attr:`other` must be
#' a real number, otherwise it should be an integer.
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
#' Each element of the tensor :attr:`other` is multiplied by the scalar
#' :attr:`alpha` and added to each element of the tensor :attr:`input`.
#' The resulting tensor is returned.
#' 
#' The shapes of :attr:`input` and :attr:`other` must be
#' :ref:`broadcastable <broadcasting-semantics>`.
#' 
#' .. math::
#'     \text{out} = \text{input} + \text{alpha} \times \text{other}
#' 
#' If :attr:`other` is of type FloatTensor or DoubleTensor, :attr:`alpha` must be
#' a real number, otherwise it should be an integer.
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
#' addmv(beta=1, input, alpha=1, mat, vec, out=None) -> Tensor
#' 
#' Performs a matrix-vector product of the matrix :attr:`mat` and
#' the vector :attr:`vec`.
#' The vector :attr:`input` is added to the final result.
#' 
#' If :attr:`mat` is a :math:`(n \times m)` tensor, :attr:`vec` is a 1-D tensor of
#' size `m`, then :attr:`input` must be
#' :ref:`broadcastable <broadcasting-semantics>` with a 1-D tensor of size `n` and
#' :attr:`out` will be 1-D tensor of size `n`.
#' 
#' :attr:`alpha` and :attr:`beta` are scaling factors on matrix-vector product between
#' :attr:`mat` and :attr:`vec` and the added tensor :attr:`input` respectively.
#' 
#' .. math::
#'     \text{out} = \beta\ \text{input} + \alpha\ (\text{mat} \mathbin{@} \text{vec})
#' 
#' For inputs of type `FloatTensor` or `DoubleTensor`, arguments :attr:`beta` and
#' :attr:`alpha` must be real numbers, otherwise they should be integers
#'
#' @param beta (Number, optional) multiplier for :attr:`input` (:math:`\beta`)
#' @param input (Tensor) vector to be added
#' @param alpha (Number, optional) multiplier for :math:`mat @ vec` (:math:`\alpha`)
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
#' addr(beta=1, input, alpha=1, vec1, vec2, out=None) -> Tensor
#' 
#' Performs the outer-product of vectors :attr:`vec1` and :attr:`vec2`
#' and adds it to the matrix :attr:`input`.
#' 
#' Optional values :attr:`beta` and :attr:`alpha` are scaling factors on the
#' outer product between :attr:`vec1` and :attr:`vec2` and the added matrix
#' :attr:`input` respectively.
#' 
#' .. math::
#'     \text{out} = \beta\ \text{input} + \alpha\ (\text{vec1} \otimes \text{vec2})
#' 
#' If :attr:`vec1` is a vector of size `n` and :attr:`vec2` is a vector
#' of size `m`, then :attr:`input` must be
#' :ref:`broadcastable <broadcasting-semantics>` with a matrix of size
#' :math:`(n \times m)` and :attr:`out` will be a matrix of size
#' :math:`(n \times m)`.
#' 
#' For inputs of type `FloatTensor` or `DoubleTensor`, arguments :attr:`beta` and
#' :attr:`alpha` must be real numbers, otherwise they should be integers
#'
#' @param beta (Number, optional) multiplier for :attr:`input` (:math:`\beta`)
#' @param input (Tensor) matrix to be added
#' @param alpha (Number, optional) multiplier for :math:`\text{vec1} \otimes \text{vec2}` (:math:`\alpha`)
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
#' allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False) -> bool
#' 
#' This function checks if all :attr:`input` and :attr:`other` satisfy the condition:
#' 
#' .. math::
#'     \lvert \text{input} - \text{other} \rvert \leq \texttt{atol} + \texttt{rtol} \times \lvert \text{other} \rvert
#' 
#' elementwise, for all elements of :attr:`input` and :attr:`other`. The behaviour of this function is analogous to
#' `numpy.allclose <https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html>`_
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
#' arange(start=0, end, step=1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
#' 
#' Returns a 1-D tensor of size :math:`\left\lceil \frac{\text{end} - \text{start}}{\text{step}} \right\rceil`
#' with values from the interval ``[start, end)`` taken with common difference
#' :attr:`step` beginning from `start`.
#' 
#' Note that non-integer :attr:`step` is subject to floating point rounding errors when
#' comparing against :attr:`end`; to avoid inconsistency, we advise adding a small epsilon to :attr:`end`
#' in such cases.
#' 
#' .. math::
#'     \text{out}_{{i+1}} = \text{out}_{i} + \text{step}
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
#' argmax(input) -> LongTensor
#' 
#' Returns the indices of the maximum value of all elements in the :attr:`input` tensor.
#' 
#' This is the second value returned by :meth:`torch.max`. See its
#' documentation for the exact semantics of this method.
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
#' argmax(input, dim, keepdim=False) -> LongTensor
#' 
#' Returns the indices of the maximum values of a tensor across a dimension.
#' 
#' This is the second value returned by :meth:`torch.max`. See its
#' documentation for the exact semantics of this method.
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
#' argmin(input) -> LongTensor
#' 
#' Returns the indices of the minimum value of all elements in the :attr:`input` tensor.
#' 
#' This is the second value returned by :meth:`torch.min`. See its
#' documentation for the exact semantics of this method.
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
#' argmin(input, dim, keepdim=False, out=None) -> LongTensor
#' 
#' Returns the indices of the minimum values of a tensor across a dimension.
#' 
#' This is the second value returned by :meth:`torch.min`. See its
#' documentation for the exact semantics of this method.
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
#' as_strided(input, size, stride, storage_offset=0) -> Tensor
#' 
#' Create a view of an existing `torch.Tensor` :attr:`input` with specified
#' :attr:`size`, :attr:`stride` and :attr:`storage_offset`.
#' 
#' .. warning::
#'     More than one element of a created tensor may refer to a single memory
#'     location. As a result, in-place operations (especially ones that are
#'     vectorized) may result in incorrect behavior. If you need to write to
#'     the tensors, please clone them first.
#' 
#'     Many PyTorch functions, which return a view of a tensor, are internally
#'     implemented with this function. Those functions, like
#'     :meth:`torch.Tensor.expand`, are easier to read and are therefore more
#'     advisable to use.
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
#' asin(input, out=None) -> Tensor
#' 
#' Returns a new tensor with the arcsine  of the elements of :attr:`input`.
#' 
#' .. math::
#'     \text{out}_{i} = \sin^{-1}(\text{input}_{i})
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
#' atan(input, out=None) -> Tensor
#' 
#' Returns a new tensor with the arctangent  of the elements of :attr:`input`.
#' 
#' .. math::
#'     \text{out}_{i} = \tan^{-1}(\text{input}_{i})
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
#' baddbmm(beta=1, input, alpha=1, batch1, batch2, out=None) -> Tensor
#' 
#' Performs a batch matrix-matrix product of matrices in :attr:`batch1`
#' and :attr:`batch2`.
#' :attr:`input` is added to the final result.
#' 
#' :attr:`batch1` and :attr:`batch2` must be 3-D tensors each containing the same
#' number of matrices.
#' 
#' If :attr:`batch1` is a :math:`(b \times n \times m)` tensor, :attr:`batch2` is a
#' :math:`(b \times m \times p)` tensor, then :attr:`input` must be
#' :ref:`broadcastable <broadcasting-semantics>` with a
#' :math:`(b \times n \times p)` tensor and :attr:`out` will be a
#' :math:`(b \times n \times p)` tensor. Both :attr:`alpha` and :attr:`beta` mean the
#' same as the scaling factors used in :meth:`torch.addbmm`.
#' 
#' .. math::
#'     \text{out}_i = \beta\ \text{input}_i + \alpha\ (\text{batch1}_i \mathbin{@} \text{batch2}_i)
#' 
#' For inputs of type `FloatTensor` or `DoubleTensor`, arguments :attr:`beta` and
#' :attr:`alpha` must be real numbers, otherwise they should be integers.
#'
#' @param beta (Number, optional) multiplier for :attr:`input` (:math:`\beta`)
#' @param input (Tensor) the tensor to be added
#' @param alpha (Number, optional) multiplier for :math:`\text{batch1} \mathbin{@} \text{batch2}` (:math:`\alpha`)
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
#' bartlett_window(window_length, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
#' 
#' Bartlett window function.
#' 
#' .. math::
#'     w[n] = 1 - \left| \frac{2n}{N-1} - 1 \right| = \begin{cases}
#'         \frac{2n}{N - 1} & \text{if } 0 \leq n \leq \frac{N - 1}{2} \\
#'         2 - \frac{2n}{N - 1} & \text{if } \frac{N - 1}{2} < n < N \\
#'     \end{cases},
#' 
#' where :math:`N` is the full window size.
#' 
#' The input :attr:`window_length` is a positive integer controlling the
#' returned window size. :attr:`periodic` flag determines whether the returned
#' window trims off the last duplicate value from the symmetric window and is
#' ready to be used as a periodic window with functions like
#' :meth:`torch.stft`. Therefore, if :attr:`periodic` is true, the :math:`N` in
#' above formula is in fact :math:`\text{window\_length} + 1`. Also, we always have
#' ``torch.bartlett_window(L, periodic=True)`` equal to
#' ``torch.bartlett_window(L + 1, periodic=False)[:-1])``.
#' 
#' .. note::
#'     If :attr:`window_length` :math:`=1`, the returned window contains a single value 1.
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
#' bernoulli(input, *, generator=None, out=None) -> Tensor
#' 
#' Draws binary random numbers (0 or 1) from a Bernoulli distribution.
#' 
#' The :attr:`input` tensor should be a tensor containing probabilities
#' to be used for drawing the binary random number.
#' Hence, all values in :attr:`input` have to be in the range:
#' :math:`0 \leq \text{input}_i \leq 1`.
#' 
#' The :math:`\text{i}^{th}` element of the output tensor will draw a
#' value :math:`1` according to the :math:`\text{i}^{th}` probability value given
#' in :attr:`input`.
#' 
#' .. math::
#'     \text{out}_{i} \sim \mathrm{Bernoulli}(p = \text{input}_{i})
#' 
#' The returned :attr:`out` tensor only has values 0 or 1 and is of the same
#' shape as :attr:`input`.
#' 
#' :attr:`out` can have integral ``dtype``, but :attr:`input` must have floating
#' point ``dtype``.
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
#' bincount(input, weights=None, minlength=0) -> Tensor
#' 
#' Count the frequency of each value in an array of non-negative ints.
#' 
#' The number of bins (size 1) is one larger than the largest value in
#' :attr:`input` unless :attr:`input` is empty, in which case the result is a
#' tensor of size 0. If :attr:`minlength` is specified, the number of bins is at least
#' :attr:`minlength` and if :attr:`input` is empty, then the result is tensor of size
#' :attr:`minlength` filled with zeros. If ``n`` is the value at position ``i``,
#' ``out[n] += weights[i]`` if :attr:`weights` is specified else
#' ``out[n] += 1``.
#' 
#' .. include:: cuda_deterministic.rst
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
#' bitwise_not(input, out=None) -> Tensor
#' 
#' Computes the bitwise NOT of the given input tensor. The input tensor must be of
#' integral or Boolean types. For bool tensors, it computes the logical NOT.
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
#' logical_not(input, out=None) -> Tensor
#' 
#' Computes the element-wise logical NOT of the given input tensor. If not specified, the output tensor will have the bool
#' dtype. If the input tensor is not a bool tensor, zeros are treated as ``False`` and non-zeros are treated as ``True``.
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
#' logical_xor(input, other, out=None) -> Tensor
#' 
#' Computes the element-wise logical XOR of the given input tensors. Zeros are treated as ``False`` and nonzeros are
#' treated as ``True``.
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
#' blackman_window(window_length, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
#' 
#' Blackman window function.
#' 
#' .. math::
#'     w[n] = 0.42 - 0.5 \cos \left( \frac{2 \pi n}{N - 1} \right) + 0.08 \cos \left( \frac{4 \pi n}{N - 1} \right)
#' 
#' where :math:`N` is the full window size.
#' 
#' The input :attr:`window_length` is a positive integer controlling the
#' returned window size. :attr:`periodic` flag determines whether the returned
#' window trims off the last duplicate value from the symmetric window and is
#' ready to be used as a periodic window with functions like
#' :meth:`torch.stft`. Therefore, if :attr:`periodic` is true, the :math:`N` in
#' above formula is in fact :math:`\text{window\_length} + 1`. Also, we always have
#' ``torch.blackman_window(L, periodic=True)`` equal to
#' ``torch.blackman_window(L + 1, periodic=False)[:-1])``.
#' 
#' .. note::
#'     If :attr:`window_length` :math:`=1`, the returned window contains a single value 1.
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
#' bmm(input, mat2, out=None) -> Tensor
#' 
#' Performs a batch matrix-matrix product of matrices stored in :attr:`input`
#' and :attr:`mat2`.
#' 
#' :attr:`input` and :attr:`mat2` must be 3-D tensors each containing
#' the same number of matrices.
#' 
#' If :attr:`input` is a :math:`(b \times n \times m)` tensor, :attr:`mat2` is a
#' :math:`(b \times m \times p)` tensor, :attr:`out` will be a
#' :math:`(b \times n \times p)` tensor.
#' 
#' .. math::
#'     \text{out}_i = \text{input}_i \mathbin{@} \text{mat2}_i
#' 
#' .. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.
#'           For broadcasting matrix products, see :func:`torch.matmul`.
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
#' broadcast_tensors(*tensors) -> List of Tensors
#' 
#'     Broadcasts the given tensors according to :ref:`broadcasting-semantics`.
#' 
#'     Args:
#'         *tensors: any number of tensors of the same type
#' 
#'     .. warning::
#' 
#'         More than one element of a broadcasted tensor may refer to a single
#'         memory location. As a result, in-place operations (especially ones that
#'         are vectorized) may result in incorrect behavior. If you need to write
#'         to the tensors, please clone them first.
#' 
#'     Example::
#' 
#'         >>> x = torch.arange(3).view(1, 3)
#'         >>> y = torch.arange(2).view(2, 1)
#'         >>> a, b = torch.broadcast_tensors(x, y)
#'         >>> a.size()
#'         torch.Size([2, 3])
#'         >>> a
#'         tensor([[0, 1, 2],
#'                 [0, 1, 2]])
#'

#'
#' 
#'
#' @name torch_broadcast_tensors
#'
#' @export
NULL


#' Cat
#'
#' cat(tensors, dim=0, out=None) -> Tensor
#' 
#' Concatenates the given sequence of :attr:`seq` tensors in the given dimension.
#' All tensors must either have the same shape (except in the concatenating
#' dimension) or be empty.
#' 
#' :func:`torch.cat` can be seen as an inverse operation for :func:`torch.split`
#' and :func:`torch.chunk`.
#' 
#' :func:`torch.cat` can be best understood via examples.
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
#' ceil(input, out=None) -> Tensor
#' 
#' Returns a new tensor with the ceil of the elements of :attr:`input`,
#' the smallest integer greater than or equal to each element.
#' 
#' .. math::
#'     \text{out}_{i} = \left\lceil \text{input}_{i} \right\rceil = \left\lfloor \text{input}_{i} \right\rfloor + 1
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
#' Returns the matrix product of the :math:`N` 2-D tensors. This product is efficiently computed
#'     using the matrix chain order algorithm which selects the order in which incurs the lowest cost in terms
#'     of arithmetic operations (`[CLRS]`_). Note that since this is a function to compute the product, :math:`N`
#'     needs to be greater than or equal to 2; if equal to 2 then a trivial matrix-matrix product is returned.
#'     If :math:`N` is 1, then this is a no-op - the original matrix is returned as is.
#' 
#' 
#'     Args:
#'         matrices (Tensors...): a sequence of 2 or more 2-D tensors whose product is to be determined.
#' 
#' 
#'     Returns:
#'         Tensor: if the :math:`i^{th}` tensor was of dimensions :math:`p_{i} \times p_{i + 1}`, then the product
#'         would be of dimensions :math:`p_{1} \times p_{N + 1}`.
#' 
#'     Example::
#' 
#'         >>> a = torch.randn(3, 4)
#'         >>> b = torch.randn(4, 5)
#'         >>> c = torch.randn(5, 6)
#'         >>> d = torch.randn(6, 7)
#'         >>> torch.chain_matmul(a, b, c, d)
#'         tensor([[ -2.3375,  -3.9790,  -4.1119,  -6.6577,   9.5609, -11.5095,  -3.2614],
#'                 [ 21.4038,   3.3378,  -8.4982,  -5.2457, -10.2561,  -2.4684,   2.7163],
#'                 [ -0.9647,  -5.8917,  -2.3213,  -5.2284,  12.8615, -12.2816,  -2.5095]])
#' 
#'     .. _`[CLRS]`: https://mitpress.mit.edu/books/introduction-algorithms-third-edition
#'

#'
#' 
#'
#' @name torch_chain_matmul
#'
#' @export
NULL


#' Chunk
#'
#' chunk(input, chunks, dim=0) -> List of Tensors
#' 
#' Splits a tensor into a specific number of chunks.
#' 
#' Last chunk will be smaller if the tensor size along the given dimension
#' :attr:`dim` is not divisible by :attr:`chunks`.
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
#' clamp(input, min, max, out=None) -> Tensor
#' 
#' Clamp all elements in :attr:`input` into the range `[` :attr:`min`, :attr:`max` `]` and return
#' a resulting tensor:
#' 
#' .. math::
#'     y_i = \begin{cases}
#'         \text{min} & \text{if } x_i < \text{min} \\
#'         x_i & \text{if } \text{min} \leq x_i \leq \text{max} \\
#'         \text{max} & \text{if } x_i > \text{max}
#'     \end{cases}
#' 
#' If :attr:`input` is of type `FloatTensor` or `DoubleTensor`, args :attr:`min`
#' and :attr:`max` must be real numbers, otherwise they should be integers.
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
#' clamp(input, *, min, out=None) -> Tensor
#' 
#' Clamps all elements in :attr:`input` to be larger or equal :attr:`min`.
#' 
#' If :attr:`input` is of type `FloatTensor` or `DoubleTensor`, :attr:`value`
#' should be a real number, otherwise it should be an integer.
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
#' clamp(input, *, max, out=None) -> Tensor
#' 
#' Clamps all elements in :attr:`input` to be smaller or equal :attr:`max`.
#' 
#' If :attr:`input` is of type `FloatTensor` or `DoubleTensor`, :attr:`value`
#' should be a real number, otherwise it should be an integer.
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
#' conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor
#' 
#' Applies a 1D convolution over an input signal composed of several input
#' planes.
#' 
#' See :class:`~torch.nn.Conv1d` for details and output shape.
#' 
#' .. include:: cudnn_deterministic.rst
#'
#' @param input NA input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iW)`
#' @param weight NA filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kW)`
#' @param bias NA optional bias of shape :math:`(\text{out\_channels})`. Default: ``None``
#' @param stride NA the stride of the convolving kernel. Can be a single number or      a one-element tuple `(sW,)`. Default: 1
#' @param padding NA implicit paddings on both sides of the input. Can be a      single number or a one-element tuple `(padW,)`. Default: 0
#' @param dilation NA the spacing between kernel elements. Can be a single number or      a one-element tuple `(dW,)`. Default: 1
#' @param groups NA split input into groups, :math:`\text{in\_channels}` should be divisible by      the number of groups. Default: 1
#'
#' 
#'
#' @name torch_conv1d
#'
#' @export
NULL


#' Conv2d
#'
#' conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor
#' 
#' Applies a 2D convolution over an input image composed of several input
#' planes.
#' 
#' See :class:`~torch.nn.Conv2d` for details and output shape.
#' 
#' .. include:: cudnn_deterministic.rst
#'
#' @param input NA input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
#' @param weight NA filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kH , kW)`
#' @param bias NA optional bias tensor of shape :math:`(\text{out\_channels})`. Default: ``None``
#' @param stride NA the stride of the convolving kernel. Can be a single number or a      tuple `(sH, sW)`. Default: 1
#' @param padding NA implicit paddings on both sides of the input. Can be a      single number or a tuple `(padH, padW)`. Default: 0
#' @param dilation NA the spacing between kernel elements. Can be a single number or      a tuple `(dH, dW)`. Default: 1
#' @param groups NA split input into groups, :math:`\text{in\_channels}` should be divisible by the      number of groups. Default: 1
#'
#' 
#'
#' @name torch_conv2d
#'
#' @export
NULL


#' Conv3d
#'
#' conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor
#' 
#' Applies a 3D convolution over an input image composed of several input
#' planes.
#' 
#' See :class:`~torch.nn.Conv3d` for details and output shape.
#' 
#' .. include:: cudnn_deterministic.rst
#'
#' @param input NA input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iT , iH , iW)`
#' @param weight NA filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kT , kH , kW)`
#' @param bias NA optional bias tensor of shape :math:`(\text{out\_channels})`. Default: None
#' @param stride NA the stride of the convolving kernel. Can be a single number or a      tuple `(sT, sH, sW)`. Default: 1
#' @param padding NA implicit paddings on both sides of the input. Can be a      single number or a tuple `(padT, padH, padW)`. Default: 0
#' @param dilation NA the spacing between kernel elements. Can be a single number or      a tuple `(dT, dH, dW)`. Default: 1
#' @param groups NA split input into groups, :math:`\text{in\_channels}` should be divisible by      the number of groups. Default: 1
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
#' @param input NA input tensor of shape :math:`(\text{sequence length} \times batch \times \text{in\_channels})`
#' @param weight NA filter of shape (:math:`\text{kernel width} \times \text{in\_channels} \times \text{out\_channels}`)
#' @param bias NA bias of shape (:math:`\text{out\_channels}`)
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
#' conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor
#' 
#' Applies a 1D transposed convolution operator over an input signal
#' composed of several input planes, sometimes also called "deconvolution".
#' 
#' See :class:`~torch.nn.ConvTranspose1d` for details and output shape.
#' 
#' .. include:: cudnn_deterministic.rst
#'
#' @param input NA input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iW)`
#' @param weight NA filters of shape :math:`(\text{in\_channels} , \frac{\text{out\_channels}}{\text{groups}} , kW)`
#' @param bias NA optional bias of shape :math:`(\text{out\_channels})`. Default: None
#' @param stride NA the stride of the convolving kernel. Can be a single number or a      tuple ``(sW,)``. Default: 1
#' @param padding NA ``dilation * (kernel_size - 1) - padding`` zero-padding will be added to both      sides of each dimension in the input. Can be a single number or a tuple      ``(padW,)``. Default: 0
#' @param output_padding NA additional size added to one side of each dimension in the      output shape. Can be a single number or a tuple ``(out_padW)``. Default: 0
#' @param groups NA split input into groups, :math:`\text{in\_channels}` should be divisible by the      number of groups. Default: 1
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
#' conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor
#' 
#' Applies a 2D transposed convolution operator over an input image
#' composed of several input planes, sometimes also called "deconvolution".
#' 
#' See :class:`~torch.nn.ConvTranspose2d` for details and output shape.
#' 
#' .. include:: cudnn_deterministic.rst
#'
#' @param input NA input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
#' @param weight NA filters of shape :math:`(\text{in\_channels} , \frac{\text{out\_channels}}{\text{groups}} , kH , kW)`
#' @param bias NA optional bias of shape :math:`(\text{out\_channels})`. Default: None
#' @param stride NA the stride of the convolving kernel. Can be a single number or a      tuple ``(sH, sW)``. Default: 1
#' @param padding NA ``dilation * (kernel_size - 1) - padding`` zero-padding will be added to both      sides of each dimension in the input. Can be a single number or a tuple      ``(padH, padW)``. Default: 0
#' @param output_padding NA additional size added to one side of each dimension in the      output shape. Can be a single number or a tuple ``(out_padH, out_padW)``.      Default: 0
#' @param groups NA split input into groups, :math:`\text{in\_channels}` should be divisible by the      number of groups. Default: 1
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
#' conv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor
#' 
#' Applies a 3D transposed convolution operator over an input image
#' composed of several input planes, sometimes also called "deconvolution"
#' 
#' See :class:`~torch.nn.ConvTranspose3d` for details and output shape.
#' 
#' .. include:: cudnn_deterministic.rst
#'
#' @param input NA input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iT , iH , iW)`
#' @param weight NA filters of shape :math:`(\text{in\_channels} , \frac{\text{out\_channels}}{\text{groups}} , kT , kH , kW)`
#' @param bias NA optional bias of shape :math:`(\text{out\_channels})`. Default: None
#' @param stride NA the stride of the convolving kernel. Can be a single number or a      tuple ``(sT, sH, sW)``. Default: 1
#' @param padding NA ``dilation * (kernel_size - 1) - padding`` zero-padding will be added to both      sides of each dimension in the input. Can be a single number or a tuple      ``(padT, padH, padW)``. Default: 0
#' @param output_padding NA additional size added to one side of each dimension in the      output shape. Can be a single number or a tuple      ``(out_padT, out_padH, out_padW)``. Default: 0
#' @param groups NA split input into groups, :math:`\text{in\_channels}` should be divisible by the      number of groups. Default: 1
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
#' cos(input, out=None) -> Tensor
#' 
#' Returns a new tensor with the cosine  of the elements of :attr:`input`.
#' 
#' .. math::
#'     \text{out}_{i} = \cos(\text{input}_{i})
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
#' cosh(input, out=None) -> Tensor
#' 
#' Returns a new tensor with the hyperbolic cosine  of the elements of
#' :attr:`input`.
#' 
#' .. math::
#'     \text{out}_{i} = \cosh(\text{input}_{i})
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
#' cumsum(input, dim, out=None, dtype=None) -> Tensor
#' 
#' Returns the cumulative sum of elements of :attr:`input` in the dimension
#' :attr:`dim`.
#' 
#' For example, if :attr:`input` is a vector of size N, the result will also be
#' a vector of size N, with elements.
#' 
#' .. math::
#'     y_i = x_1 + x_2 + x_3 + \dots + x_i
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
#' cumprod(input, dim, out=None, dtype=None) -> Tensor
#' 
#' Returns the cumulative product of elements of :attr:`input` in the dimension
#' :attr:`dim`.
#' 
#' For example, if :attr:`input` is a vector of size N, the result will also be
#' a vector of size N, with elements.
#' 
#' .. math::
#'     y_i = x_1 \times x_2\times x_3\times \dots \times x_i
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
#' det(input) -> Tensor
#' 
#' Calculates determinant of a square matrix or batches of square matrices.
#' 
#' .. note::
#'     Backward through :meth:`det` internally uses SVD results when :attr:`input` is
#'     not invertible. In this case, double backward through :meth:`det` will be
#'     unstable in when :attr:`input` doesn't have distinct singular values. See
#'     :meth:`~torch.svd` for details.
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
#' diag_embed(input, offset=0, dim1=-2, dim2=-1) -> Tensor
#' 
#' Creates a tensor whose diagonals of certain 2D planes (specified by
#' :attr:`dim1` and :attr:`dim2`) are filled by :attr:`input`.
#' To facilitate creating batched diagonal matrices, the 2D planes formed by
#' the last two dimensions of the returned tensor are chosen by default.
#' 
#' The argument :attr:`offset` controls which diagonal to consider:
#' 
#' - If :attr:`offset` = 0, it is the main diagonal.
#' - If :attr:`offset` > 0, it is above the main diagonal.
#' - If :attr:`offset` < 0, it is below the main diagonal.
#' 
#' The size of the new matrix will be calculated to make the specified diagonal
#' of the size of the last input dimension.
#' Note that for :attr:`offset` other than :math:`0`, the order of :attr:`dim1`
#' and :attr:`dim2` matters. Exchanging them is equivalent to changing the
#' sign of :attr:`offset`.
#' 
#' Applying :meth:`torch.diagonal` to the output of this function with
#' the same arguments yields a matrix identical to input. However,
#' :meth:`torch.diagonal` has different default dimensions, so those
#' need to be explicitly specified.
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
#' diagflat(input, offset=0) -> Tensor
#' 
#' - If :attr:`input` is a vector (1-D tensor), then returns a 2-D square tensor
#'   with the elements of :attr:`input` as the diagonal.
#' - If :attr:`input` is a tensor with more than one dimension, then returns a
#'   2-D tensor with diagonal elements equal to a flattened :attr:`input`.
#' 
#' The argument :attr:`offset` controls which diagonal to consider:
#' 
#' - If :attr:`offset` = 0, it is the main diagonal.
#' - If :attr:`offset` > 0, it is above the main diagonal.
#' - If :attr:`offset` < 0, it is below the main diagonal.
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
#' diagonal(input, offset=0, dim1=0, dim2=1) -> Tensor
#' 
#' Returns a partial view of :attr:`input` with the its diagonal elements
#' with respect to :attr:`dim1` and :attr:`dim2` appended as a dimension
#' at the end of the shape.
#' 
#' The argument :attr:`offset` controls which diagonal to consider:
#' 
#' - If :attr:`offset` = 0, it is the main diagonal.
#' - If :attr:`offset` > 0, it is above the main diagonal.
#' - If :attr:`offset` < 0, it is below the main diagonal.
#' 
#' Applying :meth:`torch.diag_embed` to the output of this function with
#' the same arguments yields a diagonal matrix with the diagonal entries
#' of the input. However, :meth:`torch.diag_embed` has different default
#' dimensions, so those need to be explicitly specified.
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
#' div(input, other, out=None) -> Tensor
#' 
#' Divides each element of the input ``input`` with the scalar ``other`` and
#' returns a new resulting tensor.
#' 
#' .. math::
#'     \text{out}_i = \frac{\text{input}_i}{\text{other}}
#' 
#' If the :class:`torch.dtype` of ``input`` and ``other`` differ, the
#' :class:`torch.dtype` of the result tensor is determined following rules
#' described in the type promotion :ref:`documentation <type-promotion-doc>`. If
#' ``out`` is specified, the result must be :ref:`castable <type-promotion-doc>`
#' to the :class:`torch.dtype` of the specified output tensor. Integral division
#' by zero leads to undefined behavior.
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
#' div(input, other, out=None) -> Tensor
#' 
#' Each element of the tensor ``input`` is divided by each element of the tensor
#' ``other``. The resulting tensor is returned.
#' 
#' .. math::
#'     \text{out}_i = \frac{\text{input}_i}{\text{other}_i}
#' 
#' The shapes of ``input`` and ``other`` must be :ref:`broadcastable
#' <broadcasting-semantics>`. If the :class:`torch.dtype` of ``input`` and
#' ``other`` differ, the :class:`torch.dtype` of the result tensor is determined
#' following rules described in the type promotion :ref:`documentation
#' <type-promotion-doc>`. If ``out`` is specified, the result must be
#' :ref:`castable <type-promotion-doc>` to the :class:`torch.dtype` of the
#' specified output tensor. Integral division by zero leads to undefined behavior.
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
#' dot(input, tensor) -> Tensor
#' 
#' Computes the dot product (inner product) of two tensors.
#' 
#' .. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.
#' 
#' Example::
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
#' einsum(equation, *operands) -> Tensor
#' 
#' This function provides a way of computing multilinear expressions (i.e. sums of products) using the
#' Einstein summation convention.
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
#' empty(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False) -> Tensor
#' 
#' Returns a tensor filled with uninitialized data. The shape of the tensor is
#' defined by the variable argument :attr:`size`.
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
#' empty_like(input, dtype=None, layout=None, device=None, requires_grad=False) -> Tensor
#' 
#' Returns an uninitialized tensor with the same size as :attr:`input`.
#' ``torch.empty_like(input)`` is equivalent to
#' ``torch.empty(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)``.
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
#' empty_strided(size, stride, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False) -> Tensor
#' 
#' Returns a tensor filled with uninitialized data. The shape and strides of the tensor is
#' defined by the variable argument :attr:`size` and :attr:`stride` respectively.
#' ``torch.empty_strided(size, stride)`` is equivalent to
#' ``torch.empty(size).as_strided(size, stride)``.
#' 
#' .. warning::
#'     More than one element of the created tensor may refer to a single memory
#'     location. As a result, in-place operations (especially ones that are
#'     vectorized) may result in incorrect behavior. If you need to write to
#'     the tensors, please clone them first.
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
#' erf(input, out=None) -> Tensor
#' 
#' Computes the error function of each element. The error function is defined as follows:
#' 
#' .. math::
#'     \mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2} dt
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
#' erfc(input, out=None) -> Tensor
#' 
#' Computes the complementary error function of each element of :attr:`input`.
#' The complementary error function is defined as follows:
#' 
#' .. math::
#'     \mathrm{erfc}(x) = 1 - \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2} dt
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
#' exp(input, out=None) -> Tensor
#' 
#' Returns a new tensor with the exponential of the elements
#' of the input tensor :attr:`input`.
#' 
#' .. math::
#'     y_{i} = e^{x_{i}}
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
#' expm1(input, out=None) -> Tensor
#' 
#' Returns a new tensor with the exponential of the elements minus 1
#' of :attr:`input`.
#' 
#' .. math::
#'     y_{i} = e^{x_{i}} - 1
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
#' eye(n, m=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
#' 
#' Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.
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
#' flatten(input, start_dim=0, end_dim=-1) -> Tensor
#' 
#' Flattens a contiguous range of dims in a tensor.
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
#' floor(input, out=None) -> Tensor
#' 
#' Returns a new tensor with the floor of the elements of :attr:`input`,
#' the largest integer less than or equal to each element.
#' 
#' .. math::
#'     \text{out}_{i} = \left\lfloor \text{input}_{i} \right\rfloor
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
#' frac(input, out=None) -> Tensor
#' 
#' Computes the fractional portion of each element in :attr:`input`.
#' 
#' .. math::
#'     \text{out}_{i} = \text{input}_{i} - \left\lfloor |\text{input}_{i}| \right\rfloor * \operatorname{sgn}(\text{input}_{i})
#' 
#' Example::
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
#' full(size, fill_value, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
#' 
#' Returns a tensor of size :attr:`size` filled with :attr:`fill_value`.
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
#' full_like(input, fill_value, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
#' 
#' Returns a tensor with the same size as :attr:`input` filled with :attr:`fill_value`.
#' ``torch.full_like(input, fill_value)`` is equivalent to
#' ``torch.full(input.size(), fill_value, dtype=input.dtype, layout=input.layout, device=input.device)``.
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
#' hann_window(window_length, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
#' 
#' Hann window function.
#' 
#' .. math::
#'     w[n] = \frac{1}{2}\ \left[1 - \cos \left( \frac{2 \pi n}{N - 1} \right)\right] =
#'             \sin^2 \left( \frac{\pi n}{N - 1} \right),
#' 
#' where :math:`N` is the full window size.
#' 
#' The input :attr:`window_length` is a positive integer controlling the
#' returned window size. :attr:`periodic` flag determines whether the returned
#' window trims off the last duplicate value from the symmetric window and is
#' ready to be used as a periodic window with functions like
#' :meth:`torch.stft`. Therefore, if :attr:`periodic` is true, the :math:`N` in
#' above formula is in fact :math:`\text{window\_length} + 1`. Also, we always have
#' ``torch.hann_window(L, periodic=True)`` equal to
#' ``torch.hann_window(L + 1, periodic=False)[:-1])``.
#' 
#' .. note::
#'     If :attr:`window_length` :math:`=1`, the returned window contains a single value 1.
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
#' hamming_window(window_length, periodic=True, alpha=0.54, beta=0.46, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
#' 
#' Hamming window function.
#' 
#' .. math::
#'     w[n] = \alpha - \beta\ \cos \left( \frac{2 \pi n}{N - 1} \right),
#' 
#' where :math:`N` is the full window size.
#' 
#' The input :attr:`window_length` is a positive integer controlling the
#' returned window size. :attr:`periodic` flag determines whether the returned
#' window trims off the last duplicate value from the symmetric window and is
#' ready to be used as a periodic window with functions like
#' :meth:`torch.stft`. Therefore, if :attr:`periodic` is true, the :math:`N` in
#' above formula is in fact :math:`\text{window\_length} + 1`. Also, we always have
#' ``torch.hamming_window(L, periodic=True)`` equal to
#' ``torch.hamming_window(L + 1, periodic=False)[:-1])``.
#' 
#' .. note::
#'     If :attr:`window_length` :math:`=1`, the returned window contains a single value 1.
#' 
#' .. note::
#'     This is a generalized version of :meth:`torch.hann_window`.
#'
#' @param window_length (int) the size of returned window
#' @param periodic (bool, optional) If True, returns a window to be used as periodic        function. If False, return a symmetric window.
#' @param alpha (float, optional) The coefficient :math:`\alpha` in the equation above
#' @param beta (float, optional) The coefficient :math:`\beta` in the equation above
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
#' ger(input, vec2, out=None) -> Tensor
#' 
#' Outer product of :attr:`input` and :attr:`vec2`.
#' If :attr:`input` is a vector of size :math:`n` and :attr:`vec2` is a vector of
#' size :math:`m`, then :attr:`out` must be a matrix of size :math:`(n \times m)`.
#' 
#' .. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.
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
#' fft(input, signal_ndim, normalized=False) -> Tensor
#' 
#' Complex-to-complex Discrete Fourier Transform
#' 
#' This method computes the complex-to-complex discrete Fourier transform.
#' Ignoring the batch dimensions, it computes the following expression:
#' 
#' .. math::
#'     X[\omega_1, \dots, \omega_d] =
#'         \sum_{n_1=0}^{N_1-1} \dots \sum_{n_d=0}^{N_d-1} x[n_1, \dots, n_d]
#'          e^{-j\ 2 \pi \sum_{i=0}^d \frac{\omega_i n_i}{N_i}},
#' 
#' where :math:`d` = :attr:`signal_ndim` is number of dimensions for the
#' signal, and :math:`N_i` is the size of signal dimension :math:`i`.
#' 
#' This method supports 1D, 2D and 3D complex-to-complex transforms, indicated
#' by :attr:`signal_ndim`. :attr:`input` must be a tensor with last dimension
#' of size 2, representing the real and imaginary components of complex
#' numbers, and should have at least ``signal_ndim + 1`` dimensions with optionally
#' arbitrary number of leading batch dimensions. If :attr:`normalized` is set to
#' ``True``, this normalizes the result by dividing it with
#' :math:`\sqrt{\prod_{i=1}^K N_i}` so that the operator is unitary.
#' 
#' Returns the real and the imaginary parts together as one tensor of the same
#' shape of :attr:`input`.
#' 
#' The inverse of this function is :func:`~torch.ifft`.
#' 
#' .. note::
#'     For CUDA tensors, an LRU cache is used for cuFFT plans to speed up
#'     repeatedly running FFT methods on tensors of same geometry with same
#'     configuration. See :ref:`cufft-plan-cache` for more details on how to
#'     monitor and control the cache.
#' 
#' .. warning::
#'     For CPU tensors, this method is currently only available with MKL. Use
#'     :func:`torch.backends.mkl.is_available` to check if MKL is installed.
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
#' ifft(input, signal_ndim, normalized=False) -> Tensor
#' 
#' Complex-to-complex Inverse Discrete Fourier Transform
#' 
#' This method computes the complex-to-complex inverse discrete Fourier
#' transform. Ignoring the batch dimensions, it computes the following
#' expression:
#' 
#' .. math::
#'     X[\omega_1, \dots, \omega_d] =
#'         \frac{1}{\prod_{i=1}^d N_i} \sum_{n_1=0}^{N_1-1} \dots \sum_{n_d=0}^{N_d-1} x[n_1, \dots, n_d]
#'          e^{\ j\ 2 \pi \sum_{i=0}^d \frac{\omega_i n_i}{N_i}},
#' 
#' where :math:`d` = :attr:`signal_ndim` is number of dimensions for the
#' signal, and :math:`N_i` is the size of signal dimension :math:`i`.
#' 
#' The argument specifications are almost identical with :func:`~torch.fft`.
#' However, if :attr:`normalized` is set to ``True``, this instead returns the
#' results multiplied by :math:`\sqrt{\prod_{i=1}^d N_i}`, to become a unitary
#' operator. Therefore, to invert a :func:`~torch.fft`, the :attr:`normalized`
#' argument should be set identically for :func:`~torch.fft`.
#' 
#' Returns the real and the imaginary parts together as one tensor of the same
#' shape of :attr:`input`.
#' 
#' The inverse of this function is :func:`~torch.fft`.
#' 
#' .. note::
#'     For CUDA tensors, an LRU cache is used for cuFFT plans to speed up
#'     repeatedly running FFT methods on tensors of same geometry with same
#'     configuration. See :ref:`cufft-plan-cache` for more details on how to
#'     monitor and control the cache.
#' 
#' .. warning::
#'     For CPU tensors, this method is currently only available with MKL. Use
#'     :func:`torch.backends.mkl.is_available` to check if MKL is installed.
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
#' rfft(input, signal_ndim, normalized=False, onesided=True) -> Tensor
#' 
#' Real-to-complex Discrete Fourier Transform
#' 
#' This method computes the real-to-complex discrete Fourier transform. It is
#' mathematically equivalent with :func:`~torch.fft` with differences only in
#' formats of the input and output.
#' 
#' This method supports 1D, 2D and 3D real-to-complex transforms, indicated
#' by :attr:`signal_ndim`. :attr:`input` must be a tensor with at least
#' ``signal_ndim`` dimensions with optionally arbitrary number of leading batch
#' dimensions. If :attr:`normalized` is set to ``True``, this normalizes the result
#' by dividing it with :math:`\sqrt{\prod_{i=1}^K N_i}` so that the operator is
#' unitary, where :math:`N_i` is the size of signal dimension :math:`i`.
#' 
#' The real-to-complex Fourier transform results follow conjugate symmetry:
#' 
#' .. math::
#'     X[\omega_1, \dots, \omega_d] = X^*[N_1 - \omega_1, \dots, N_d - \omega_d],
#' 
#' where the index arithmetic is computed modulus the size of the corresponding
#' dimension, :math:`\ ^*` is the conjugate operator, and
#' :math:`d` = :attr:`signal_ndim`. :attr:`onesided` flag controls whether to avoid
#' redundancy in the output results. If set to ``True`` (default), the output will
#' not be full complex result of shape :math:`(*, 2)`, where :math:`*` is the shape
#' of :attr:`input`, but instead the last dimension will be halfed as of size
#' :math:`\lfloor \frac{N_d}{2} \rfloor + 1`.
#' 
#' The inverse of this function is :func:`~torch.irfft`.
#' 
#' .. note::
#'     For CUDA tensors, an LRU cache is used for cuFFT plans to speed up
#'     repeatedly running FFT methods on tensors of same geometry with same
#'     configuration. See :ref:`cufft-plan-cache` for more details on how to
#'     monitor and control the cache.
#' 
#' .. warning::
#'     For CPU tensors, this method is currently only available with MKL. Use
#'     :func:`torch.backends.mkl.is_available` to check if MKL is installed.
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
#' irfft(input, signal_ndim, normalized=False, onesided=True, signal_sizes=None) -> Tensor
#' 
#' Complex-to-real Inverse Discrete Fourier Transform
#' 
#' This method computes the complex-to-real inverse discrete Fourier transform.
#' It is mathematically equivalent with :func:`ifft` with differences only in
#' formats of the input and output.
#' 
#' The argument specifications are almost identical with :func:`~torch.ifft`.
#' Similar to :func:`~torch.ifft`, if :attr:`normalized` is set to ``True``,
#' this normalizes the result by multiplying it with
#' :math:`\sqrt{\prod_{i=1}^K N_i}` so that the operator is unitary, where
#' :math:`N_i` is the size of signal dimension :math:`i`.
#' 
#' .. note::
#'     Due to the conjugate symmetry, :attr:`input` do not need to contain the full
#'     complex frequency values. Roughly half of the values will be sufficient, as
#'     is the case when :attr:`input` is given by :func:`~torch.rfft` with
#'     ``rfft(signal, onesided=True)``. In such case, set the :attr:`onesided`
#'     argument of this method to ``True``. Moreover, the original signal shape
#'     information can sometimes be lost, optionally set :attr:`signal_sizes` to be
#'     the size of the original signal (without the batch dimensions if in batched
#'     mode) to recover it with correct shape.
#' 
#'     Therefore, to invert an :func:`~torch.rfft`, the :attr:`normalized` and
#'     :attr:`onesided` arguments should be set identically for :func:`~torch.irfft`,
#'     and preferrably a :attr:`signal_sizes` is given to avoid size mismatch. See the
#'     example below for a case of size mismatch.
#' 
#'     See :func:`~torch.rfft` for details on conjugate symmetry.
#' 
#' The inverse of this function is :func:`~torch.rfft`.
#' 
#' .. warning::
#'     Generally speaking, input to this function should contain values
#'     following conjugate symmetry. Note that even if :attr:`onesided` is
#'     ``True``, often symmetry on some part is still needed. When this
#'     requirement is not satisfied, the behavior of :func:`~torch.irfft` is
#'     undefined. Since :func:`torch.autograd.gradcheck` estimates numerical
#'     Jacobian with point perturbations, :func:`~torch.irfft` will almost
#'     certainly fail the check.
#' 
#' .. note::
#'     For CUDA tensors, an LRU cache is used for cuFFT plans to speed up
#'     repeatedly running FFT methods on tensors of same geometry with same
#'     configuration. See :ref:`cufft-plan-cache` for more details on how to
#'     monitor and control the cache.
#' 
#' .. warning::
#'     For CPU tensors, this method is currently only available with MKL. Use
#'     :func:`torch.backends.mkl.is_available` to check if MKL is installed.
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
#' inverse(input, out=None) -> Tensor
#' 
#' Takes the inverse of the square matrix :attr:`input`. :attr:`input` can be batches
#' of 2D square tensors, in which case this function would return a tensor composed of
#' individual inverses.
#' 
#' .. note::
#' 
#'     Irrespective of the original strides, the returned tensors will be
#'     transposed, i.e. with strides like `input.contiguous().transpose(-2, -1).stride()`
#'
#' @param input (Tensor) the input tensor of size :math:`(*, n, n)` where `*` is zero or more                    batch dimensions
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
#' is_floating_point(input) -> (bool)
#' 
#' Returns True if the data type of :attr:`input` is a floating point data type i.e.,
#' one of ``torch.float64``, ``torch.float32`` and ``torch.float16``.
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
#' kthvalue(input, k, dim=None, keepdim=False, out=None) -> (Tensor, LongTensor)
#' 
#' Returns a namedtuple ``(values, indices)`` where ``values`` is the :attr:`k` th
#' smallest element of each row of the :attr:`input` tensor in the given dimension
#' :attr:`dim`. And ``indices`` is the index location of each element found.
#' 
#' If :attr:`dim` is not given, the last dimension of the `input` is chosen.
#' 
#' If :attr:`keepdim` is ``True``, both the :attr:`values` and :attr:`indices` tensors
#' are the same size as :attr:`input`, except in the dimension :attr:`dim` where
#' they are of size 1. Otherwise, :attr:`dim` is squeezed
#' (see :func:`torch.squeeze`), resulting in both the :attr:`values` and
#' :attr:`indices` tensors having 1 fewer dimension than the :attr:`input` tensor.
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
#' linspace(start, end, steps=100, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
#' 
#' Returns a one-dimensional tensor of :attr:`steps`
#' equally spaced points between :attr:`start` and :attr:`end`.
#' 
#' The output tensor is 1-D of size :attr:`steps`.
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
#' log(input, out=None) -> Tensor
#' 
#' Returns a new tensor with the natural logarithm of the elements
#' of :attr:`input`.
#' 
#' .. math::
#'     y_{i} = \log_{e} (x_{i})
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
#' log10(input, out=None) -> Tensor
#' 
#' Returns a new tensor with the logarithm to the base 10 of the elements
#' of :attr:`input`.
#' 
#' .. math::
#'     y_{i} = \log_{10} (x_{i})
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
#' log1p(input, out=None) -> Tensor
#' 
#' Returns a new tensor with the natural logarithm of (1 + :attr:`input`).
#' 
#' .. math::
#'     y_i = \log_{e} (x_i + 1)
#' 
#' .. note:: This function is more accurate than :func:`torch.log` for small
#'           values of :attr:`input`
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
#' log2(input, out=None) -> Tensor
#' 
#' Returns a new tensor with the logarithm to the base 2 of the elements
#' of :attr:`input`.
#' 
#' .. math::
#'     y_{i} = \log_{2} (x_{i})
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
#' logdet(input) -> Tensor
#' 
#' Calculates log determinant of a square matrix or batches of square matrices.
#' 
#' .. note::
#'     Result is ``-inf`` if :attr:`input` has zero log determinant, and is ``nan`` if
#'     :attr:`input` has negative determinant.
#' 
#' .. note::
#'     Backward through :meth:`logdet` internally uses SVD results when :attr:`input`
#'     is not invertible. In this case, double backward through :meth:`logdet` will
#'     be unstable in when :attr:`input` doesn't have distinct singular values. See
#'     :meth:`~torch.svd` for details.
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
#' logspace(start, end, steps=100, base=10.0, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
#' 
#' Returns a one-dimensional tensor of :attr:`steps` points
#' logarithmically spaced with base :attr:`base` between
#' :math:`{\text{base}}^{\text{start}}` and :math:`{\text{base}}^{\text{end}}`.
#' 
#' The output tensor is 1-D of size :attr:`steps`.
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
#' Returns the log of summed exponentials of each row of the :attr:`input`
#' tensor in the given dimension :attr:`dim`. The computation is numerically
#' stabilized.
#' 
#' For summation index :math:`j` given by `dim` and other indices :math:`i`, the result is
#' 
#'     .. math::
#'         \text{logsumexp}(x)_{i} = \log \sum_j \exp(x_{ij})
#' 
#' 
#' If :attr:`keepdim` is ``True``, the output tensor is of the same size
#' as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
#' Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
#' output tensor having 1 (or ``len(dim)``) fewer dimension(s).
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
#' matmul(input, other, out=None) -> Tensor
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
#'   must be broadcastable).  For example, if :attr:`input` is a
#'   :math:`(j \times 1 \times n \times m)` tensor and :attr:`other` is a :math:`(k \times m \times p)`
#'   tensor, :attr:`out` will be an :math:`(j \times k \times n \times p)` tensor.
#' 
#' .. note::
#' 
#'     The 1-dimensional dot product version of this function does not support an :attr:`out` parameter.
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
#' matrix_rank(input, tol=None, symmetric=False) -> Tensor
#' 
#' Returns the numerical rank of a 2-D tensor. The method to compute the
#' matrix rank is done using SVD by default. If :attr:`symmetric` is ``True``,
#' then :attr:`input` is assumed to be symmetric, and the computation of the
#' rank is done by obtaining the eigenvalues.
#' 
#' :attr:`tol` is the threshold below which the singular values (or the eigenvalues
#' when :attr:`symmetric` is ``True``) are considered to be 0. If :attr:`tol` is not
#' specified, :attr:`tol` is set to ``S.max() * max(S.size()) * eps`` where `S` is the
#' singular values (or the eigenvalues when :attr:`symmetric` is ``True``), and ``eps``
#' is the epsilon value for the datatype of :attr:`input`.
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
#' matrix_power(input, n) -> Tensor
#' 
#' Returns the matrix raised to the power :attr:`n` for square matrices.
#' For batch of matrices, each individual matrix is raised to the power :attr:`n`.
#' 
#' If :attr:`n` is negative, then the inverse of the matrix (if invertible) is
#' raised to the power :attr:`n`.  For a batch of matrices, the batched inverse
#' (if invertible) is raised to the power :attr:`n`. If :attr:`n` is 0, then an identity matrix
#' is returned.
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
#' max(input) -> Tensor
#' 
#' Returns the maximum value of all elements in the :attr:`input` tensor.
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
#' max(input, dim, keepdim=False, out=None) -> (Tensor, LongTensor)
#' 
#' Returns a namedtuple ``(values, indices)`` where ``values`` is the maximum
#' value of each row of the :attr:`input` tensor in the given dimension
#' :attr:`dim`. And ``indices`` is the index location of each maximum value found
#' (argmax).
#' 
#' If :attr:`keepdim` is ``True``, the output tensors are of the same size
#' as :attr:`input` except in the dimension :attr:`dim` where they are of size 1.
#' Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting
#' in the output tensors having 1 fewer dimension than :attr:`input`.
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
#' max(input, other, out=None) -> Tensor
#' 
#' Each element of the tensor :attr:`input` is compared with the corresponding
#' element of the tensor :attr:`other` and an element-wise maximum is taken.
#' 
#' The shapes of :attr:`input` and :attr:`other` don't need to match,
#' but they must be :ref:`broadcastable <broadcasting-semantics>`.
#' 
#' .. math::
#'     \text{out}_i = \max(\text{tensor}_i, \text{other}_i)
#' 
#' .. note:: When the shapes do not match, the shape of the returned output tensor
#'           follows the :ref:`broadcasting rules <broadcasting-semantics>`.
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
#' mean(input) -> Tensor
#' 
#' Returns the mean value of all elements in the :attr:`input` tensor.
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
#' mean(input, dim, keepdim=False, out=None) -> Tensor
#' 
#' Returns the mean value of each row of the :attr:`input` tensor in the given
#' dimension :attr:`dim`. If :attr:`dim` is a list of dimensions,
#' reduce over all of them.
#' 
#' 
#' If :attr:`keepdim` is ``True``, the output tensor is of the same size
#' as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
#' Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
#' output tensor having 1 (or ``len(dim)``) fewer dimension(s).
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
#' median(input) -> Tensor
#' 
#' Returns the median value of all elements in the :attr:`input` tensor.
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
#' median(input, dim=-1, keepdim=False, values=None, indices=None) -> (Tensor, LongTensor)
#' 
#' Returns a namedtuple ``(values, indices)`` where ``values`` is the median
#' value of each row of the :attr:`input` tensor in the given dimension
#' :attr:`dim`. And ``indices`` is the index location of each median value found.
#' 
#' By default, :attr:`dim` is the last dimension of the :attr:`input` tensor.
#' 
#' If :attr:`keepdim` is ``True``, the output tensors are of the same size
#' as :attr:`input` except in the dimension :attr:`dim` where they are of size 1.
#' Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in
#' the outputs tensor having 1 fewer dimension than :attr:`input`.
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
#' min(input) -> Tensor
#' 
#' Returns the minimum value of all elements in the :attr:`input` tensor.
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
#' min(input, dim, keepdim=False, out=None) -> (Tensor, LongTensor)
#' 
#' Returns a namedtuple ``(values, indices)`` where ``values`` is the minimum
#' value of each row of the :attr:`input` tensor in the given dimension
#' :attr:`dim`. And ``indices`` is the index location of each minimum value found
#' (argmin).
#' 
#' If :attr:`keepdim` is ``True``, the output tensors are of the same size as
#' :attr:`input` except in the dimension :attr:`dim` where they are of size 1.
#' Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in
#' the output tensors having 1 fewer dimension than :attr:`input`.
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
#' min(input, other, out=None) -> Tensor
#' 
#' Each element of the tensor :attr:`input` is compared with the corresponding
#' element of the tensor :attr:`other` and an element-wise minimum is taken.
#' The resulting tensor is returned.
#' 
#' The shapes of :attr:`input` and :attr:`other` don't need to match,
#' but they must be :ref:`broadcastable <broadcasting-semantics>`.
#' 
#' .. math::
#'     \text{out}_i = \min(\text{tensor}_i, \text{other}_i)
#' 
#' .. note:: When the shapes do not match, the shape of the returned output tensor
#'           follows the :ref:`broadcasting rules <broadcasting-semantics>`.
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
#' mm(input, mat2, out=None) -> Tensor
#' 
#' Performs a matrix multiplication of the matrices :attr:`input` and :attr:`mat2`.
#' 
#' If :attr:`input` is a :math:`(n \times m)` tensor, :attr:`mat2` is a
#' :math:`(m \times p)` tensor, :attr:`out` will be a :math:`(n \times p)` tensor.
#' 
#' .. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.
#'           For broadcasting matrix products, see :func:`torch.matmul`.
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
#' mode(input, dim=-1, keepdim=False, values=None, indices=None) -> (Tensor, LongTensor)
#' 
#' Returns a namedtuple ``(values, indices)`` where ``values`` is the mode
#' value of each row of the :attr:`input` tensor in the given dimension
#' :attr:`dim`, i.e. a value which appears most often
#' in that row, and ``indices`` is the index location of each mode value found.
#' 
#' By default, :attr:`dim` is the last dimension of the :attr:`input` tensor.
#' 
#' If :attr:`keepdim` is ``True``, the output tensors are of the same size as
#' :attr:`input` except in the dimension :attr:`dim` where they are of size 1.
#' Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting
#' in the output tensors having 1 fewer dimension than :attr:`input`.
#' 
#' .. note:: This function is not defined for ``torch.cuda.Tensor`` yet.
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
#' Multiplies each element of the input :attr:`input` with the scalar
#' :attr:`other` and returns a new resulting tensor.
#' 
#' .. math::
#'     \text{out}_i = \text{other} \times \text{input}_i
#' 
#' If :attr:`input` is of type `FloatTensor` or `DoubleTensor`, :attr:`other`
#' should be a real number, otherwise it should be an integer
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
#' Each element of the tensor :attr:`input` is multiplied by the corresponding
#' element of the Tensor :attr:`other`. The resulting tensor is returned.
#' 
#' The shapes of :attr:`input` and :attr:`other` must be
#' :ref:`broadcastable <broadcasting-semantics>`.
#' 
#' .. math::
#'     \text{out}_i = \text{input}_i \times \text{other}_i
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
#' mv(input, vec, out=None) -> Tensor
#' 
#' Performs a matrix-vector product of the matrix :attr:`input` and the vector
#' :attr:`vec`.
#' 
#' If :attr:`input` is a :math:`(n \times m)` tensor, :attr:`vec` is a 1-D tensor of
#' size :math:`m`, :attr:`out` will be 1-D of size :math:`n`.
#' 
#' .. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.
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
#' mvlgamma(input, p) -> Tensor
#' 
#' Computes the multivariate log-gamma function (`[reference]`_) with dimension :math:`p` element-wise, given by
#' 
#' .. math::
#'     \log(\Gamma_{p}(a)) = C + \displaystyle \sum_{i=1}^{p} \log\left(\Gamma\left(a - \frac{i - 1}{2}\right)\right)
#' 
#' where :math:`C = \log(\pi) \times \frac{p (p - 1)}{4}` and :math:`\Gamma(\cdot)` is the Gamma function.
#' 
#' If any of the elements are less than or equal to :math:`\frac{p - 1}{2}`, then an error
#' is thrown.
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
#' narrow(input, dim, start, length) -> Tensor
#' 
#' Returns a new tensor that is a narrowed version of :attr:`input` tensor. The
#' dimension :attr:`dim` is input from :attr:`start` to :attr:`start + length`. The
#' returned tensor and :attr:`input` tensor share the same underlying storage.
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
#' ones(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
#' 
#' Returns a tensor filled with the scalar value `1`, with the shape defined
#' by the variable argument :attr:`size`.
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
#' ones_like(input, dtype=None, layout=None, device=None, requires_grad=False) -> Tensor
#' 
#' Returns a tensor filled with the scalar value `1`, with the same size as
#' :attr:`input`. ``torch.ones_like(input)`` is equivalent to
#' ``torch.ones(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)``.
#' 
#' .. warning::
#'     As of 0.4, this function does not support an :attr:`out` keyword. As an alternative,
#'     the old ``torch.ones_like(input, out=output)`` is equivalent to
#'     ``torch.ones(input.size(), out=output)``.
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
#'     Args:
#'         x1 (Tensor): input tensor of shape :math:`B \times P \times M`.
#'         x2 (Tensor): input tensor of shape :math:`B \times R \times M`.
#'         p: p value for the p-norm distance to calculate between each vector pair
#'             :math:`\in [0, \infty]`.
#'         compute_mode:
#'             'use_mm_for_euclid_dist_if_necessary' - will use matrix multiplication approach to calculate
#'             euclidean distance (p = 2) if P > 25 or R > 25
#'             'use_mm_for_euclid_dist' - will always use matrix multiplication approach to calculate
#'             euclidean distance (p = 2)
#'             'donot_use_mm_for_euclid_dist' - will never use matrix multiplication approach to calculate
#'             euclidean distance (p = 2)
#'             Default: use_mm_for_euclid_dist_if_necessary.
#' 
#'     If x1 has shape :math:`B \times P \times M` and x2 has shape :math:`B \times R \times M` then the
#'     output will have shape :math:`B \times P \times R`.
#' 
#'     This function is equivalent to `scipy.spatial.distance.cdist(input,'minkowski', p=p)`
#'     if :math:`p \in (0, \infty)`. When :math:`p = 0` it is equivalent to
#'     `scipy.spatial.distance.cdist(input, 'hamming') * M`. When :math:`p = \infty`, the closest
#'     scipy function is `scipy.spatial.distance.cdist(xn, lambda x, y: np.abs(x - y).max())`.
#' 
#'     Example:
#' 
#'         >>> a = torch.tensor([[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059]])
#'         >>> a
#'         tensor([[ 0.9041,  0.0196],
#'                 [-0.3108, -2.4423],
#'                 [-0.4821,  1.0590]])
#'         >>> b = torch.tensor([[-2.1763, -0.4713], [-0.6986,  1.3702]])
#'         >>> b
#'         tensor([[-2.1763, -0.4713],
#'                 [-0.6986,  1.3702]])
#'         >>> torch.cdist(a, b, p=2)
#'         tensor([[3.1193, 2.0959],
#'                 [2.7138, 3.8322],
#'                 [2.2830, 0.3791]])
#'

#'
#' 
#'
#' @name torch_cdist
#'
#' @export
NULL


#' Pdist
#'
#' pdist(input, p=2) -> Tensor
#' 
#' Computes the p-norm distance between every pair of row vectors in the input.
#' This is identical to the upper triangular portion, excluding the diagonal, of
#' `torch.norm(input[:, None] - input, dim=2, p=p)`. This function will be faster
#' if the rows are contiguous.
#' 
#' If input has shape :math:`N \times M` then the output will have shape
#' :math:`\frac{1}{2} N (N - 1)`.
#' 
#' This function is equivalent to `scipy.spatial.distance.pdist(input,
#' 'minkowski', p=p)` if :math:`p \in (0, \infty)`. When :math:`p = 0` it is
#' equivalent to `scipy.spatial.distance.pdist(input, 'hamming') * M`.
#' When :math:`p = \infty`, the closest scipy function is
#' `scipy.spatial.distance.pdist(xn, lambda x, y: np.abs(x - y).max())`.
#'
#' @param input NA input tensor of shape :math:`N \times M`.
#' @param p NA p value for the p-norm distance to calculate between each vector pair        :math:`\in [0, \infty]`.
#'
#' 
#'
#' @name torch_pdist
#'
#' @export
NULL


#' Cosine_similarity
#'
#' cosine_similarity(x1, x2, dim=1, eps=1e-8) -> Tensor
#' 
#' Returns cosine similarity between x1 and x2, computed along dim.
#' 
#' .. math ::
#'     \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}
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
#' Rearranges elements in a tensor of shape :math:`(*, C \times r^2, H, W)` to a
#' tensor of shape :math:`(*, C, H \times r, W \times r)`.
#' 
#' See :class:`~torch.nn.PixelShuffle` for details.
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
#' pinverse(input, rcond=1e-15) -> Tensor
#' 
#' Calculates the pseudo-inverse (also known as the Moore-Penrose inverse) of a 2D tensor.
#' Please look at `Moore-Penrose inverse`_ for more details
#' 
#' .. note::
#'     This method is implemented using the Singular Value Decomposition.
#' 
#' .. note::
#'     The pseudo-inverse is not necessarily a continuous function in the elements of the matrix `[1]`_.
#'     Therefore, derivatives are not always existent, and exist for a constant rank only `[2]`_.
#'     However, this method is backprop-able due to the implementation by using SVD results, and
#'     could be unstable. Double-backward will also be unstable due to the usage of SVD internally.
#'     See :meth:`~torch.svd` for more details.
#'
#' @param input (Tensor) The input tensor of size :math:`(*, m, n)` where :math:`*` is zero or more batch dimensions
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
#' rand(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
#' 
#' Returns a tensor filled with random numbers from a uniform distribution
#' on the interval :math:`[0, 1)`
#' 
#' The shape of the tensor is defined by the variable argument :attr:`size`.
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
#' rand_like(input, dtype=None, layout=None, device=None, requires_grad=False) -> Tensor
#' 
#' Returns a tensor with the same size as :attr:`input` that is filled with
#' random numbers from a uniform distribution on the interval :math:`[0, 1)`.
#' ``torch.rand_like(input)`` is equivalent to
#' ``torch.rand(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)``.
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
#' randint(low=0, high, size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
#' 
#' Returns a tensor filled with random integers generated uniformly
#' between :attr:`low` (inclusive) and :attr:`high` (exclusive).
#' 
#' The shape of the tensor is defined by the variable argument :attr:`size`.
#' 
#' .. note:
#'     With the global dtype default (``torch.float32``), this function returns
#'     a tensor with dtype ``torch.int64``.
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
#' randint_like(input, low=0, high, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
#' 
#' Returns a tensor with the same shape as Tensor :attr:`input` filled with
#' random integers generated uniformly between :attr:`low` (inclusive) and
#' :attr:`high` (exclusive).
#' 
#' .. note:
#'     With the global dtype default (``torch.float32``), this function returns
#'     a tensor with dtype ``torch.int64``.
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
#' randn(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
#' 
#' Returns a tensor filled with random numbers from a normal distribution
#' with mean `0` and variance `1` (also called the standard normal
#' distribution).
#' 
#' .. math::
#'     \text{out}_{i} \sim \mathcal{N}(0, 1)
#' 
#' The shape of the tensor is defined by the variable argument :attr:`size`.
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
#' randn_like(input, dtype=None, layout=None, device=None, requires_grad=False) -> Tensor
#' 
#' Returns a tensor with the same size as :attr:`input` that is filled with
#' random numbers from a normal distribution with mean 0 and variance 1.
#' ``torch.randn_like(input)`` is equivalent to
#' ``torch.randn(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)``.
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
#' randperm(n, out=None, dtype=torch.int64, layout=torch.strided, device=None, requires_grad=False) -> LongTensor
#' 
#' Returns a random permutation of integers from ``0`` to ``n - 1``.
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
#' range(start=0, end, step=1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
#' 
#' Returns a 1-D tensor of size :math:`\left\lfloor \frac{\text{end} - \text{start}}{\text{step}} \right\rfloor + 1`
#' with values from :attr:`start` to :attr:`end` with step :attr:`step`. Step is
#' the gap between two values in the tensor.
#' 
#' .. math::
#'     \text{out}_{i+1} = \text{out}_i + \text{step}.
#' 
#' .. warning::
#'     This function is deprecated in favor of :func:`torch.arange`.
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
#' reciprocal(input, out=None) -> Tensor
#' 
#' Returns a new tensor with the reciprocal of the elements of :attr:`input`
#' 
#' .. math::
#'     \text{out}_{i} = \frac{1}{\text{input}_{i}}
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
#' neg(input, out=None) -> Tensor
#' 
#' Returns a new tensor with the negative of the elements of :attr:`input`.
#' 
#' .. math::
#'     \text{out} = -1 \times \text{input}
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
#' repeat_interleave(input, repeats, dim=None) -> Tensor
#' 
#' Repeat elements of a tensor.
#' 
#' .. warning::
#' 
#'     This is different from :func:`torch.repeat` but similar to `numpy.repeat`.
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
#' repeat_interleave(repeats) -> Tensor
#' 
#' If the `repeats` is `tensor([n1, n2, n3, ...])`, then the output will be
#' `tensor([0, 0, ..., 1, 1, ..., 2, 2, ..., ...])` where `0` appears `n1` times,
#' `1` appears `n2` times, `2` appears `n3` times, etc.
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
#' reshape(input, shape) -> Tensor
#' 
#' Returns a tensor with the same data and number of elements as :attr:`input`,
#' but with the specified shape. When possible, the returned tensor will be a view
#' of :attr:`input`. Otherwise, it will be a copy. Contiguous inputs and inputs
#' with compatible strides can be reshaped without copying, but you should not
#' depend on the copying vs. viewing behavior.
#' 
#' See :meth:`torch.Tensor.view` on when it is possible to return a view.
#' 
#' A single dimension may be -1, in which case it's inferred from the remaining
#' dimensions and the number of elements in :attr:`input`.
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
#' round(input, out=None) -> Tensor
#' 
#' Returns a new tensor with each of the elements of :attr:`input` rounded
#' to the closest integer.
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
#' rrelu_(input, lower=1./8, upper=1./3, training=False) -> Tensor
#' 
#' In-place version of :func:`~rrelu`.
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
#' relu_(input) -> Tensor
#' 
#' In-place version of :func:`~relu`.
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
#' rsqrt(input, out=None) -> Tensor
#' 
#' Returns a new tensor with the reciprocal of the square-root of each of
#' the elements of :attr:`input`.
#' 
#' .. math::
#'     \text{out}_{i} = \frac{1}{\sqrt{\text{input}_{i}}}
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
#' selu_(input) -> Tensor
#' 
#' In-place version of :func:`~selu`.
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
#' celu_(input, alpha=1.) -> Tensor
#' 
#' In-place version of :func:`~celu`.
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
#' sigmoid(input, out=None) -> Tensor
#' 
#' Returns a new tensor with the sigmoid of the elements of :attr:`input`.
#' 
#' .. math::
#'     \text{out}_{i} = \frac{1}{1 + e^{-\text{input}_{i}}}
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
#' sin(input, out=None) -> Tensor
#' 
#' Returns a new tensor with the sine of the elements of :attr:`input`.
#' 
#' .. math::
#'     \text{out}_{i} = \sin(\text{input}_{i})
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
#' sinh(input, out=None) -> Tensor
#' 
#' Returns a new tensor with the hyperbolic sine of the elements of
#' :attr:`input`.
#' 
#' .. math::
#'     \text{out}_{i} = \sinh(\text{input}_{i})
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
#' slogdet(input) -> (Tensor, Tensor)
#' 
#' Calculates the sign and log absolute value of the determinant(s) of a square matrix or batches of square matrices.
#' 
#' .. note::
#'     If ``input`` has zero determinant, this returns ``(0, -inf)``.
#' 
#' .. note::
#'     Backward through :meth:`slogdet` internally uses SVD results when :attr:`input`
#'     is not invertible. In this case, double backward through :meth:`slogdet`
#'     will be unstable in when :attr:`input` doesn't have distinct singular values.
#'     See :meth:`~torch.svd` for details.
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
#'     If :attr:`split_size_or_sections` is an integer type, then :attr:`tensor` will
#'     be split into equally sized chunks (if possible). Last chunk will be smaller if
#'     the tensor size along the given dimension :attr:`dim` is not divisible by
#'     :attr:`split_size`.
#' 
#'     If :attr:`split_size_or_sections` is a list, then :attr:`tensor` will be split
#'     into ``len(split_size_or_sections)`` chunks with sizes in :attr:`dim` according
#'     to :attr:`split_size_or_sections`.
#' 
#'     Arguments:
#'         tensor (Tensor): tensor to split.
#'         split_size_or_sections (int) or (list(int)): size of a single chunk or
#'             list of sizes for each chunk
#'         dim (int): dimension along which to split the tensor.
#'

#'
#' 
#'
#' @name torch_split
#'
#' @export
NULL


#' Squeeze
#'
#' squeeze(input, dim=None, out=None) -> Tensor
#' 
#' Returns a tensor with all the dimensions of :attr:`input` of size `1` removed.
#' 
#' For example, if `input` is of shape:
#' :math:`(A \times 1 \times B \times C \times 1 \times D)` then the `out` tensor
#' will be of shape: :math:`(A \times B \times C \times D)`.
#' 
#' When :attr:`dim` is given, a squeeze operation is done only in the given
#' dimension. If `input` is of shape: :math:`(A \times 1 \times B)`,
#' ``squeeze(input, 0)`` leaves the tensor unchanged, but ``squeeze(input, 1)``
#' will squeeze the tensor to the shape :math:`(A \times B)`.
#' 
#' .. note:: The returned tensor shares the storage with the input tensor,
#'           so changing the contents of one will change the contents of the other.
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
#' stack(tensors, dim=0, out=None) -> Tensor
#' 
#' Concatenates sequence of tensors along a new dimension.
#' 
#' All tensors need to be of the same size.
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
#'     .. math::
#'         X[m, \omega] = \sum_{k = 0}^{\text{win\_length-1}}%
#'                             \text{window}[k]\ \text{input}[m \times \text{hop\_length} + k]\ %
#'                             \exp\left(- j \frac{2 \pi \cdot \omega k}{\text{win\_length}}\right),
#' 
#'     where :math:`m` is the index of the sliding window, and :math:`\omega` is
#'     the frequency that :math:`0 \leq \omega < \text{n\_fft}`. When
#'     :attr:`onesided` is the default value ``True``,
#' 
#'     * :attr:`input` must be either a 1-D time sequence or a 2-D batch of time
#'       sequences.
#' 
#'     * If :attr:`hop_length` is ``None`` (default), it is treated as equal to
#'       ``floor(n_fft / 4)``.
#' 
#'     * If :attr:`win_length` is ``None`` (default), it is treated as equal to
#'       :attr:`n_fft`.
#' 
#'     * :attr:`window` can be a 1-D tensor of size :attr:`win_length`, e.g., from
#'       :meth:`torch.hann_window`. If :attr:`window` is ``None`` (default), it is
#'       treated as if having :math:`1` everywhere in the window. If
#'       :math:`\text{win\_length} < \text{n\_fft}`, :attr:`window` will be padded on
#'       both sides to length :attr:`n_fft` before being applied.
#' 
#'     * If :attr:`center` is ``True`` (default), :attr:`input` will be padded on
#'       both sides so that the :math:`t`-th frame is centered at time
#'       :math:`t \times \text{hop\_length}`. Otherwise, the :math:`t`-th frame
#'       begins at time  :math:`t \times \text{hop\_length}`.
#' 
#'     * :attr:`pad_mode` determines the padding method used on :attr:`input` when
#'       :attr:`center` is ``True``. See :meth:`torch.nn.functional.pad` for
#'       all available options. Default is ``"reflect"``.
#' 
#'     * If :attr:`onesided` is ``True`` (default), only values for :math:`\omega`
#'       in :math:`\left[0, 1, 2, \dots, \left\lfloor \frac{\text{n\_fft}}{2} \right\rfloor + 1\right]`
#'       are returned because the real-to-complex Fourier transform satisfies the
#'       conjugate symmetry, i.e., :math:`X[m, \omega] = X[m, \text{n\_fft} - \omega]^*`.
#' 
#'     * If :attr:`normalized` is ``True`` (default is ``False``), the function
#'       returns the normalized STFT results, i.e., multiplied by :math:`(\text{frame\_length})^{-0.5}`.
#' 
#'     Returns the real and the imaginary parts together as one tensor of size
#'     :math:`(* \times N \times T \times 2)`, where :math:`*` is the optional
#'     batch size of :attr:`input`, :math:`N` is the number of frequencies where
#'     STFT is applied, :math:`T` is the total number of frames used, and each pair
#'     in the last dimension represents a complex number as the real part and the
#'     imaginary part.
#' 
#'     .. warning::
#'       This function changed signature at version 0.4.1. Calling with the
#'       previous signature may cause error or return incorrect result.
#' 
#'     Arguments:
#'         input (Tensor): the input tensor
#'         n_fft (int): size of Fourier transform
#'         hop_length (int, optional): the distance between neighboring sliding window
#'             frames. Default: ``None`` (treated as equal to ``floor(n_fft / 4)``)
#'         win_length (int, optional): the size of window frame and STFT filter.
#'             Default: ``None``  (treated as equal to :attr:`n_fft`)
#'         window (Tensor, optional): the optional window function.
#'             Default: ``None`` (treated as window of all :math:`1` s)
#'         center (bool, optional): whether to pad :attr:`input` on both sides so
#'             that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
#'             Default: ``True``
#'         pad_mode (string, optional): controls the padding method used when
#'             :attr:`center` is ``True``. Default: ``"reflect"``
#'         normalized (bool, optional): controls whether to return the normalized STFT results
#'              Default: ``False``
#'         onesided (bool, optional): controls whether to return half of results to
#'             avoid redundancy Default: ``True``
#' 
#'     Returns:
#'         Tensor: A tensor containing the STFT result with shape described above
#'

#'
#' 
#'
#' @name torch_stft
#'
#' @export
NULL


#' Sum
#'
#' sum(input, dtype=None) -> Tensor
#' 
#' Returns the sum of all elements in the :attr:`input` tensor.
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
#' sum(input, dim, keepdim=False, dtype=None) -> Tensor
#' 
#' Returns the sum of each row of the :attr:`input` tensor in the given
#' dimension :attr:`dim`. If :attr:`dim` is a list of dimensions,
#' reduce over all of them.
#' 
#' 
#' If :attr:`keepdim` is ``True``, the output tensor is of the same size
#' as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
#' Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
#' output tensor having 1 (or ``len(dim)``) fewer dimension(s).
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
#' sqrt(input, out=None) -> Tensor
#' 
#' Returns a new tensor with the square-root of the elements of :attr:`input`.
#' 
#' .. math::
#'     \text{out}_{i} = \sqrt{\text{input}_{i}}
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
#' std(input, unbiased=True) -> Tensor
#' 
#' Returns the standard-deviation of all elements in the :attr:`input` tensor.
#' 
#' If :attr:`unbiased` is ``False``, then the standard-deviation will be calculated
#' via the biased estimator. Otherwise, Bessel's correction will be used.
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
#' std(input, dim, keepdim=False, unbiased=True, out=None) -> Tensor
#' 
#' Returns the standard-deviation of each row of the :attr:`input` tensor in the
#' dimension :attr:`dim`. If :attr:`dim` is a list of dimensions,
#' reduce over all of them.
#' 
#' 
#' If :attr:`keepdim` is ``True``, the output tensor is of the same size
#' as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
#' Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
#' output tensor having 1 (or ``len(dim)``) fewer dimension(s).
#' 
#' 
#' If :attr:`unbiased` is ``False``, then the standard-deviation will be calculated
#' via the biased estimator. Otherwise, Bessel's correction will be used.
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
#' std_mean(input, unbiased=True) -> (Tensor, Tensor)
#' 
#' Returns the standard-deviation and mean of all elements in the :attr:`input` tensor.
#' 
#' If :attr:`unbiased` is ``False``, then the standard-deviation will be calculated
#' via the biased estimator. Otherwise, Bessel's correction will be used.
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
#' std(input, dim, keepdim=False, unbiased=True) -> (Tensor, Tensor)
#' 
#' Returns the standard-deviation and mean of each row of the :attr:`input` tensor in the
#' dimension :attr:`dim`. If :attr:`dim` is a list of dimensions,
#' reduce over all of them.
#' 
#' 
#' If :attr:`keepdim` is ``True``, the output tensor is of the same size
#' as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
#' Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
#' output tensor having 1 (or ``len(dim)``) fewer dimension(s).
#' 
#' 
#' If :attr:`unbiased` is ``False``, then the standard-deviation will be calculated
#' via the biased estimator. Otherwise, Bessel's correction will be used.
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
#' prod(input, dtype=None) -> Tensor
#' 
#' Returns the product of all elements in the :attr:`input` tensor.
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
#' prod(input, dim, keepdim=False, dtype=None) -> Tensor
#' 
#' Returns the product of each row of the :attr:`input` tensor in the given
#' dimension :attr:`dim`.
#' 
#' If :attr:`keepdim` is ``True``, the output tensor is of the same size
#' as :attr:`input` except in the dimension :attr:`dim` where it is of size 1.
#' Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in
#' the output tensor having 1 fewer dimension than :attr:`input`.
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
#' t(input) -> Tensor
#' 
#' Expects :attr:`input` to be <= 2-D tensor and transposes dimensions 0
#' and 1.
#' 
#' 0-D and 1-D tensors are returned as it is and
#' 2-D tensor can be seen as a short-hand function for ``transpose(input, 0, 1)``.
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
#' tan(input, out=None) -> Tensor
#' 
#' Returns a new tensor with the tangent of the elements of :attr:`input`.
#' 
#' .. math::
#'     \text{out}_{i} = \tan(\text{input}_{i})
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
#' tanh(input, out=None) -> Tensor
#' 
#' Returns a new tensor with the hyperbolic tangent of the elements
#' of :attr:`input`.
#' 
#' .. math::
#'     \text{out}_{i} = \tanh(\text{input}_{i})
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
#'     :attr:`tensordot` implements a generalized matrix product.
#' 
#'     Args:
#'       a (Tensor): Left tensor to contract
#'       b (Tensor): Right tensor to contract
#'       dims (int or tuple of two lists of integers): number of dimensions to
#'          contract or explicit lists of dimensions for :attr:`a` and
#'          :attr:`b` respectively
#' 
#'     When called with an integer argument :attr:`dims` = :math:`d`, and the number of
#'     dimensions of :attr:`a` and :attr:`b` is :math:`m` and :math:`n`, respectively,
#'     it computes
#' 
#'     .. math::
#'         r_{i_0,...,i_{m-d}, i_d,...,i_n}
#'           = \sum_{k_0,...,k_{d-1}} a_{i_0,...,i_{m-d},k_0,...,k_{d-1}} \times b_{k_0,...,k_{d-1}, i_d,...,i_n}.
#' 
#'     When called with :attr:`dims` of the list form, the given dimensions will be contracted
#'     in place of the last :math:`d` of :attr:`a` and the first :math:`d` of :math:`b`. The sizes
#'     in these dimensions must match, but :attr:`tensordot` will deal with broadcasted
#'     dimensions.
#' 
#'     Examples::
#' 
#'         >>> a = torch.arange(60.).reshape(3, 4, 5)
#'         >>> b = torch.arange(24.).reshape(4, 3, 2)
#'         >>> torch.tensordot(a, b, dims=([1, 0], [0, 1]))
#'         tensor([[4400., 4730.],
#'                 [4532., 4874.],
#'                 [4664., 5018.],
#'                 [4796., 5162.],
#'                 [4928., 5306.]])
#' 
#'         >>> a = torch.randn(3, 4, 5, device='cuda')
#'         >>> b = torch.randn(4, 5, 6, device='cuda')
#'         >>> c = torch.tensordot(a, b, dims=2).cpu()
#'         tensor([[ 8.3504, -2.5436,  6.2922,  2.7556, -1.0732,  3.2741],
#'                 [ 3.3161,  0.0704,  5.0187, -0.4079, -4.3126,  4.8744],
#'                 [ 0.8223,  3.9445,  3.2168, -0.2400,  3.4117,  1.7780]])
#'

#'
#' 
#'
#' @name torch_tensordot
#'
#' @export
NULL


#' Threshold_
#'
#' threshold_(input, threshold, value) -> Tensor
#' 
#' In-place version of :func:`~threshold`.
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
#' transpose(input, dim0, dim1) -> Tensor
#' 
#' Returns a tensor that is a transposed version of :attr:`input`.
#' The given dimensions :attr:`dim0` and :attr:`dim1` are swapped.
#' 
#' The resulting :attr:`out` tensor shares it's underlying storage with the
#' :attr:`input` tensor, so changing the content of one would change the content
#' of the other.
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
#' flip(input, dims) -> Tensor
#' 
#' Reverse the order of a n-D tensor along given axis in dims.
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
#' roll(input, shifts, dims=None) -> Tensor
#' 
#' Roll the tensor along the given dimension(s). Elements that are shifted beyond the
#' last position are re-introduced at the first position. If a dimension is not
#' specified, the tensor will be flattened before rolling and then restored
#' to the original shape.
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
#' rot90(input, k, dims) -> Tensor
#' 
#' Rotate a n-D tensor by 90 degrees in the plane specified by dims axis.
#' Rotation direction is from the first towards the second axis if k > 0, and from the second towards the first for k < 0.
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
#' trapz(y, x, *, dim=-1) -> Tensor
#' 
#' Estimate :math:`\int y\,dx` along `dim`, using the trapezoid rule.
#'
#' @param y (Tensor) The values of the function to integrate
#' @param x (Tensor) The points at which the function `y` is sampled.        If `x` is not in ascending order, intervals on which it is decreasing        contribute negatively to the estimated integral (i.e., the convention        :math:`\int_a^b f = -\int_b^a f` is followed).
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
#' trapz(y, *, dx=1, dim=-1) -> Tensor
#' 
#' As above, but the sample points are spaced uniformly at a distance of `dx`.
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
#' trunc(input, out=None) -> Tensor
#' 
#' Returns a new tensor with the truncated integer values of
#' the elements of :attr:`input`.
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
#'     .. note:: This function is different from :func:`torch.unique` in the sense that this function
#'         only eliminates consecutive duplicate values. This semantics is similar to `std::unique`
#'         in C++.
#' 
#'     Arguments:
#'         input (Tensor): the input tensor
#'         return_inverse (bool): Whether to also return the indices for where
#'             elements in the original input ended up in the returned unique list.
#'         return_counts (bool): Whether to also return the counts for each unique
#'             element.
#'         dim (int): the dimension to apply unique. If ``None``, the unique of the
#'             flattened input is returned. default: ``None``
#' 
#'     Returns:
#'         (Tensor, Tensor (optional), Tensor (optional)): A tensor or a tuple of tensors containing
#' 
#'             - **output** (*Tensor*): the output list of unique scalar elements.
#'             - **inverse_indices** (*Tensor*): (optional) if
#'               :attr:`return_inverse` is True, there will be an additional
#'               returned tensor (same shape as input) representing the indices
#'               for where elements in the original input map to in the output;
#'               otherwise, this function will only return a single tensor.
#'             - **counts** (*Tensor*): (optional) if
#'               :attr:`return_counts` is True, there will be an additional
#'               returned tensor (same shape as output or output.size(dim),
#'               if dim was specified) representing the number of occurrences
#'               for each unique value or tensor.
#' 
#'     Example::
#' 
#'         >>> x = torch.tensor([1, 1, 2, 2, 3, 1, 1, 2])
#'         >>> output = torch.unique_consecutive(x)
#'         >>> output
#'         tensor([1, 2, 3, 1, 2])
#' 
#'         >>> output, inverse_indices = torch.unique_consecutive(x, return_inverse=True)
#'         >>> output
#'         tensor([1, 2, 3, 1, 2])
#'         >>> inverse_indices
#'         tensor([0, 0, 1, 1, 2, 3, 3, 4])
#' 
#'         >>> output, counts = torch.unique_consecutive(x, return_counts=True)
#'         >>> output
#'         tensor([1, 2, 3, 1, 2])
#'         >>> counts
#'         tensor([2, 2, 1, 2, 1])
#'

#'
#' 
#'
#' @name torch_unique_consecutive
#'
#' @export
NULL


#' Unsqueeze
#'
#' unsqueeze(input, dim, out=None) -> Tensor
#' 
#' Returns a new tensor with a dimension of size one inserted at the
#' specified position.
#' 
#' The returned tensor shares the same underlying data with this tensor.
#' 
#' A :attr:`dim` value within the range ``[-input.dim() - 1, input.dim() + 1)``
#' can be used. Negative :attr:`dim` will correspond to :meth:`unsqueeze`
#' applied at :attr:`dim` = ``dim + input.dim() + 1``.
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
#' var(input, unbiased=True) -> Tensor
#' 
#' Returns the variance of all elements in the :attr:`input` tensor.
#' 
#' If :attr:`unbiased` is ``False``, then the variance will be calculated via the
#' biased estimator. Otherwise, Bessel's correction will be used.
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
#' var(input, dim, keepdim=False, unbiased=True, out=None) -> Tensor
#' 
#' Returns the variance of each row of the :attr:`input` tensor in the given
#' dimension :attr:`dim`.
#' 
#' 
#' If :attr:`keepdim` is ``True``, the output tensor is of the same size
#' as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
#' Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
#' output tensor having 1 (or ``len(dim)``) fewer dimension(s).
#' 
#' 
#' If :attr:`unbiased` is ``False``, then the variance will be calculated via the
#' biased estimator. Otherwise, Bessel's correction will be used.
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
#' var_mean(input, unbiased=True) -> (Tensor, Tensor)
#' 
#' Returns the variance and mean of all elements in the :attr:`input` tensor.
#' 
#' If :attr:`unbiased` is ``False``, then the variance will be calculated via the
#' biased estimator. Otherwise, Bessel's correction will be used.
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
#' var_mean(input, dim, keepdim=False, unbiased=True) -> (Tensor, Tensor)
#' 
#' Returns the variance and mean of each row of the :attr:`input` tensor in the given
#' dimension :attr:`dim`.
#' 
#' 
#' If :attr:`keepdim` is ``True``, the output tensor is of the same size
#' as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
#' Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
#' output tensor having 1 (or ``len(dim)``) fewer dimension(s).
#' 
#' 
#' If :attr:`unbiased` is ``False``, then the variance will be calculated via the
#' biased estimator. Otherwise, Bessel's correction will be used.
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
#' where(condition, x, y) -> Tensor
#' 
#' Return a tensor of elements selected from either :attr:`x` or :attr:`y`, depending on :attr:`condition`.
#' 
#' The operation is defined as:
#' 
#' .. math::
#'     \text{out}_i = \begin{cases}
#'         \text{x}_i & \text{if } \text{condition}_i \\
#'         \text{y}_i & \text{otherwise} \\
#'     \end{cases}
#' 
#' .. note::
#'     The tensors :attr:`condition`, :attr:`x`, :attr:`y` must be :ref:`broadcastable <broadcasting-semantics>`.
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
#' where(condition) -> tuple of LongTensor
#' 
#' ``torch.where(condition)`` is identical to
#' ``torch.nonzero(condition, as_tuple=True)``.
#' 
#' .. note::
#'     See also :func:`torch.nonzero`.
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
#' zeros(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
#' 
#' Returns a tensor filled with the scalar value `0`, with the shape defined
#' by the variable argument :attr:`size`.
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
#' zeros_like(input, dtype=None, layout=None, device=None, requires_grad=False) -> Tensor
#' 
#' Returns a tensor filled with the scalar value `0`, with the same size as
#' :attr:`input`. ``torch.zeros_like(input)`` is equivalent to
#' ``torch.zeros(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)``.
#' 
#' .. warning::
#'     As of 0.4, this function does not support an :attr:`out` keyword. As an alternative,
#'     the old ``torch.zeros_like(input, out=output)`` is equivalent to
#'     ``torch.zeros(input.size(), out=output)``.
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
#'     Args:
#'         input (Tensor): the input tensor
#'         p (int, float, inf, -inf, 'fro', 'nuc', optional): the order of norm. Default: ``'fro'``
#'             The following norms can be calculated:
#' 
#'             =====  ============================  ==========================
#'             ord    matrix norm                   vector norm
#'             =====  ============================  ==========================
#'             None   Frobenius norm                2-norm
#'             'fro'  Frobenius norm                --
#'             'nuc'  nuclear norm                  --
#'             Other  as vec norm when dim is None  sum(abs(x)**ord)**(1./ord)
#'             =====  ============================  ==========================
#' 
#'         dim (int, 2-tuple of ints, 2-list of ints, optional): If it is an int,
#'             vector norm will be calculated, if it is 2-tuple of ints, matrix norm
#'             will be calculated. If the value is None, matrix norm will be calculated
#'             when the input tensor only has two dimensions, vector norm will be
#'             calculated when the input tensor only has one dimension. If the input
#'             tensor has more than two dimensions, the vector norm will be applied to
#'             last dimension.
#'         keepdim (bool, optional): whether the output tensors have :attr:`dim`
#'             retained or not. Ignored if :attr:`dim` = ``None`` and
#'             :attr:`out` = ``None``. Default: ``False``
#'         out (Tensor, optional): the output tensor. Ignored if
#'             :attr:`dim` = ``None`` and :attr:`out` = ``None``.
#'         dtype (:class:`torch.dtype`, optional): the desired data type of
#'             returned tensor. If specified, the input tensor is casted to
#'             :attr:'dtype' while performing the operation. Default: None.
#' 
#' 
#'     Example::
#' 
#'         >>> import torch
#'         >>> a = torch.arange(9, dtype= torch.float) - 4
#'         >>> b = a.reshape((3, 3))
#'         >>> torch.norm(a)
#'         tensor(7.7460)
#'         >>> torch.norm(b)
#'         tensor(7.7460)
#'         >>> torch.norm(a, float('inf'))
#'         tensor(4.)
#'         >>> torch.norm(b, float('inf'))
#'         tensor(4.)
#'         >>> c = torch.tensor([[ 1, 2, 3],[-1, 1, 4]] , dtype= torch.float)
#'         >>> torch.norm(c, dim=0)
#'         tensor([1.4142, 2.2361, 5.0000])
#'         >>> torch.norm(c, dim=1)
#'         tensor([3.7417, 4.2426])
#'         >>> torch.norm(c, p=1, dim=1)
#'         tensor([6., 6.])
#'         >>> d = torch.arange(8, dtype= torch.float).reshape(2,2,2)
#'         >>> torch.norm(d, dim=(1,2))
#'         tensor([ 3.7417, 11.2250])
#'         >>> torch.norm(d[0, :, :]), torch.norm(d[1, :, :])
#'         (tensor(3.7417), tensor(11.2250))
#'

#'
#' 
#'
#' @name torch_norm
#'
#' @export
NULL


#' Pow
#'
#' pow(input, exponent, out=None) -> Tensor
#' 
#' Takes the power of each element in :attr:`input` with :attr:`exponent` and
#' returns a tensor with the result.
#' 
#' :attr:`exponent` can be either a single ``float`` number or a `Tensor`
#' with the same number of elements as :attr:`input`.
#' 
#' When :attr:`exponent` is a scalar value, the operation applied is:
#' 
#' .. math::
#'     \text{out}_i = x_i ^ \text{exponent}
#' 
#' When :attr:`exponent` is a tensor, the operation applied is:
#' 
#' .. math::
#'     \text{out}_i = x_i ^ {\text{exponent}_i}
#' 
#' When :attr:`exponent` is a tensor, the shapes of :attr:`input`
#' and :attr:`exponent` must be :ref:`broadcastable <broadcasting-semantics>`.
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
#' pow(self, exponent, out=None) -> Tensor
#' 
#' :attr:`self` is a scalar ``float`` value, and :attr:`exponent` is a tensor.
#' The returned tensor :attr:`out` is of the same shape as :attr:`exponent`
#' 
#' The operation applied is:
#' 
#' .. math::
#'     \text{out}_i = \text{self} ^ {\text{exponent}_i}
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
#' addmm(beta=1, input, alpha=1, mat1, mat2, out=None) -> Tensor
#' 
#' Performs a matrix multiplication of the matrices :attr:`mat1` and :attr:`mat2`.
#' The matrix :attr:`input` is added to the final result.
#' 
#' If :attr:`mat1` is a :math:`(n \times m)` tensor, :attr:`mat2` is a
#' :math:`(m \times p)` tensor, then :attr:`input` must be
#' :ref:`broadcastable <broadcasting-semantics>` with a :math:`(n \times p)` tensor
#' and :attr:`out` will be a :math:`(n \times p)` tensor.
#' 
#' :attr:`alpha` and :attr:`beta` are scaling factors on matrix-vector product between
#' :attr:`mat1` and :attr:`mat2` and the added matrix :attr:`input` respectively.
#' 
#' .. math::
#'     \text{out} = \beta\ \text{input} + \alpha\ (\text{mat1}_i \mathbin{@} \text{mat2}_i)
#' 
#' For inputs of type `FloatTensor` or `DoubleTensor`, arguments :attr:`beta` and
#' :attr:`alpha` must be real numbers, otherwise they should be integers.
#'
#' @param beta (Number, optional) multiplier for :attr:`input` (:math:`\beta`)
#' @param input (Tensor) matrix to be added
#' @param alpha (Number, optional) multiplier for :math:`mat1 @ mat2` (:math:`\alpha`)
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
#' sparse_coo_tensor(indices, values, size=None, dtype=None, device=None, requires_grad=False) -> Tensor
#' 
#' Constructs a sparse tensors in COO(rdinate) format with non-zero elements at the given :attr:`indices`
#' with the given :attr:`values`. A sparse tensor can be `uncoalesced`, in that case, there are duplicate
#' coordinates in the indices, and the value at that index is the sum of all duplicate value entries:
#' `torch.sparse`_.
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
#' unbind(input, dim=0) -> seq
#' 
#' Removes a tensor dimension.
#' 
#' Returns a tuple of all slices along a given dimension, already without it.
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
#' quantize_per_tensor(input, scale, zero_point, dtype) -> Tensor
#' 
#' Converts a float tensor to quantized tensor with given scale and zero point.
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
#' quantize_per_channel(input, scales, zero_points, axis, dtype) -> Tensor
#' 
#' Converts a float tensor to per-channel quantized tensor with given scales and zero points.
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
#' Take :math:`N` tensors, each of which can be either scalar or 1-dimensional
#' vector, and create :math:`N` N-dimensional grids, where the :math:`i` :sup:`th` grid is defined by
#' expanding the :math:`i` :sup:`th` input over dimensions defined by other inputs.
#' 
#' 
#'     Args:
#'         tensors (list of Tensor): list of scalars or 1 dimensional tensors. Scalars will be
#'         treated as tensors of size :math:`(1,)` automatically
#' 
#'     Returns:
#'         seq (sequence of Tensors): If the input has :math:`k` tensors of size
#'         :math:`(N_1,), (N_2,), \ldots , (N_k,)`, then the output would also have :math:`k` tensors,
#'         where all tensors are of size :math:`(N_1, N_2, \ldots , N_k)`.
#' 
#'     Example::
#' 
#'         >>> x = torch.tensor([1, 2, 3])
#'         >>> y = torch.tensor([4, 5, 6])
#'         >>> grid_x, grid_y = torch.meshgrid(x, y)
#'         >>> grid_x
#'         tensor([[1, 1, 1],
#'                 [2, 2, 2],
#'                 [3, 3, 3]])
#'         >>> grid_y
#'         tensor([[4, 5, 6],
#'                 [4, 5, 6],
#'                 [4, 5, 6]])
#'

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
#'     Arguments:
#'         *tensors: any number of 1 dimensional tensors.
#' 
#'     Returns:
#'         Tensor: A tensor equivalent to converting all the input tensors into lists,
#'             do `itertools.product` on these lists, and finally convert the resulting list
#'             into tensor.
#' 
#'     Example::
#' 
#'         >>> a = [1, 2, 3]
#'         >>> b = [4, 5]
#'         >>> list(itertools.product(a, b))
#'         [(1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5)]
#'         >>> tensor_a = torch.tensor(a)
#'         >>> tensor_b = torch.tensor(b)
#'         >>> torch.cartesian_prod(tensor_a, tensor_b)
#'         tensor([[1, 4],
#'                 [1, 5],
#'                 [2, 4],
#'                 [2, 5],
#'                 [3, 4],
#'                 [3, 5]])
#'

#'
#' 
#'
#' @name torch_cartesian_prod
#'
#' @export
NULL


#' Combinations
#'
#' combinations(input, r=2, with_replacement=False) -> seq
#' 
#' Compute combinations of length :math:`r` of the given tensor. The behavior is similar to
#' python's `itertools.combinations` when `with_replacement` is set to `False`, and
#' `itertools.combinations_with_replacement` when `with_replacement` is set to `True`.
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
#' result_type(tensor1, tensor2) -> dtype
#' 
#' Returns the :class:`torch.dtype` that would result from performing an arithmetic
#' operation on the provided input tensors. See type promotion :ref:`documentation <type-promotion-doc>`
#' for more information on the type promotion logic.
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
#' can_cast(from, to) -> bool
#' 
#' Determines if a type conversion is allowed under PyTorch casting rules
#' described in the type promotion :ref:`documentation <type-promotion-doc>`.
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
#' promote_types(type1, type2) -> dtype
#' 
#' Returns the :class:`torch.dtype` with the smallest size and scalar kind that is
#' not smaller nor of lower kind than either `type1` or `type2`. See type promotion
#' :ref:`documentation <type-promotion-doc>` for more information on the type
#' promotion logic.
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
#' bitwise_xor(input, other, out=None) -> Tensor
#' 
#' Computes the bitwise XOR of :attr:`input` and :attr:`other`. The input tensor must be of
#' integral or Boolean types. For bool tensors, it computes the logical XOR.
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
#' addbmm(beta=1, input, alpha=1, batch1, batch2, out=None) -> Tensor
#' 
#' Performs a batch matrix-matrix product of matrices stored
#' in :attr:`batch1` and :attr:`batch2`,
#' with a reduced add step (all matrix multiplications get accumulated
#' along the first dimension).
#' :attr:`input` is added to the final result.
#' 
#' :attr:`batch1` and :attr:`batch2` must be 3-D tensors each containing the
#' same number of matrices.
#' 
#' If :attr:`batch1` is a :math:`(b \times n \times m)` tensor, :attr:`batch2` is a
#' :math:`(b \times m \times p)` tensor, :attr:`input` must be
#' :ref:`broadcastable <broadcasting-semantics>` with a :math:`(n \times p)` tensor
#' and :attr:`out` will be a :math:`(n \times p)` tensor.
#' 
#' .. math::
#'     out = \beta\ \text{input} + \alpha\ (\sum_{i=0}^{b-1} \text{batch1}_i \mathbin{@} \text{batch2}_i)
#' 
#' For inputs of type `FloatTensor` or `DoubleTensor`, arguments :attr:`beta` and :attr:`alpha`
#' must be real numbers, otherwise they should be integers.
#'
#' @param beta (Number, optional) multiplier for :attr:`input` (:math:`\beta`)
#' @param input (Tensor) matrix to be added
#' @param alpha (Number, optional) multiplier for `batch1 @ batch2` (:math:`\alpha`)
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
#' diag(input, diagonal=0, out=None) -> Tensor
#' 
#' - If :attr:`input` is a vector (1-D tensor), then returns a 2-D square tensor
#'   with the elements of :attr:`input` as the diagonal.
#' - If :attr:`input` is a matrix (2-D tensor), then returns a 1-D tensor with
#'   the diagonal elements of :attr:`input`.
#' 
#' The argument :attr:`diagonal` controls which diagonal to consider:
#' 
#' - If :attr:`diagonal` = 0, it is the main diagonal.
#' - If :attr:`diagonal` > 0, it is above the main diagonal.
#' - If :attr:`diagonal` < 0, it is below the main diagonal.
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
#' cross(input, other, dim=-1, out=None) -> Tensor
#' 
#' 
#' Returns the cross product of vectors in dimension :attr:`dim` of :attr:`input`
#' and :attr:`other`.
#' 
#' :attr:`input` and :attr:`other` must have the same size, and the size of their
#' :attr:`dim` dimension should be 3.
#' 
#' If :attr:`dim` is not given, it defaults to the first dimension found with the
#' size 3.
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
#' triu(input, diagonal=0, out=None) -> Tensor
#' 
#' Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices
#' :attr:`input`, the other elements of the result tensor :attr:`out` are set to 0.
#' 
#' The upper triangular part of the matrix is defined as the elements on and
#' above the diagonal.
#' 
#' The argument :attr:`diagonal` controls which diagonal to consider. If
#' :attr:`diagonal` = 0, all elements on and above the main diagonal are
#' retained. A positive value excludes just as many diagonals above the main
#' diagonal, and similarly a negative value includes just as many diagonals below
#' the main diagonal. The main diagonal are the set of indices
#' :math:`\lbrace (i, i) \rbrace` for :math:`i \in [0, \min\{d_{1}, d_{2}\} - 1]` where
#' :math:`d_{1}, d_{2}` are the dimensions of the matrix.
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
#' tril(input, diagonal=0, out=None) -> Tensor
#' 
#' Returns the lower triangular part of the matrix (2-D tensor) or batch of matrices
#' :attr:`input`, the other elements of the result tensor :attr:`out` are set to 0.
#' 
#' The lower triangular part of the matrix is defined as the elements on and
#' below the diagonal.
#' 
#' The argument :attr:`diagonal` controls which diagonal to consider. If
#' :attr:`diagonal` = 0, all elements on and below the main diagonal are
#' retained. A positive value includes just as many diagonals above the main
#' diagonal, and similarly a negative value excludes just as many diagonals below
#' the main diagonal. The main diagonal are the set of indices
#' :math:`\lbrace (i, i) \rbrace` for :math:`i \in [0, \min\{d_{1}, d_{2}\} - 1]` where
#' :math:`d_{1}, d_{2}` are the dimensions of the matrix.
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
#' tril_indices(row, col, offset=0, dtype=torch.long, device='cpu', layout=torch.strided) -> Tensor
#' 
#' Returns the indices of the lower triangular part of a :attr:`row`-by-
#' :attr:`col` matrix in a 2-by-N Tensor, where the first row contains row
#' coordinates of all indices and the second row contains column coordinates.
#' Indices are ordered based on rows and then columns.
#' 
#' The lower triangular part of the matrix is defined as the elements on and
#' below the diagonal.
#' 
#' The argument :attr:`offset` controls which diagonal to consider. If
#' :attr:`offset` = 0, all elements on and below the main diagonal are
#' retained. A positive value includes just as many diagonals above the main
#' diagonal, and similarly a negative value excludes just as many diagonals below
#' the main diagonal. The main diagonal are the set of indices
#' :math:`\lbrace (i, i) \rbrace` for :math:`i \in [0, \min\{d_{1}, d_{2}\} - 1]`
#' where :math:`d_{1}, d_{2}` are the dimensions of the matrix.
#' 
#' NOTE: when running on 'cuda', row * col must be less than :math:`2^{59}` to
#' prevent overflow during calculation.
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
#' triu_indices(row, col, offset=0, dtype=torch.long, device='cpu', layout=torch.strided) -> Tensor
#' 
#' Returns the indices of the upper triangular part of a :attr:`row` by
#' :attr:`col` matrix in a 2-by-N Tensor, where the first row contains row
#' coordinates of all indices and the second row contains column coordinates.
#' Indices are ordered based on rows and then columns.
#' 
#' The upper triangular part of the matrix is defined as the elements on and
#' above the diagonal.
#' 
#' The argument :attr:`offset` controls which diagonal to consider. If
#' :attr:`offset` = 0, all elements on and above the main diagonal are
#' retained. A positive value excludes just as many diagonals above the main
#' diagonal, and similarly a negative value includes just as many diagonals below
#' the main diagonal. The main diagonal are the set of indices
#' :math:`\lbrace (i, i) \rbrace` for :math:`i \in [0, \min\{d_{1}, d_{2}\} - 1]`
#' where :math:`d_{1}, d_{2}` are the dimensions of the matrix.
#' 
#' NOTE: when running on 'cuda', row * col must be less than :math:`2^{59}` to
#' prevent overflow during calculation.
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
#' trace(input) -> Tensor
#' 
#' Returns the sum of the elements of the diagonal of the input 2-D matrix.
#' 
#' Example::
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
#' ne(input, other, out=None) -> Tensor
#' 
#' Computes :math:`input \neq other` element-wise.
#' 
#' The second argument can be a number or a tensor whose shape is
#' :ref:`broadcastable <broadcasting-semantics>` with the first argument.
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
#' eq(input, other, out=None) -> Tensor
#' 
#' Computes element-wise equality
#' 
#' The second argument can be a number or a tensor whose shape is
#' :ref:`broadcastable <broadcasting-semantics>` with the first argument.
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
#' ge(input, other, out=None) -> Tensor
#' 
#' Computes :math:`\text{input} \geq \text{other}` element-wise.
#' 
#' The second argument can be a number or a tensor whose shape is
#' :ref:`broadcastable <broadcasting-semantics>` with the first argument.
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
#' le(input, other, out=None) -> Tensor
#' 
#' Computes :math:`\text{input} \leq \text{other}` element-wise.
#' 
#' The second argument can be a number or a tensor whose shape is
#' :ref:`broadcastable <broadcasting-semantics>` with the first argument.
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
#' gt(input, other, out=None) -> Tensor
#' 
#' Computes :math:`\text{input} > \text{other}` element-wise.
#' 
#' The second argument can be a number or a tensor whose shape is
#' :ref:`broadcastable <broadcasting-semantics>` with the first argument.
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
#' lt(input, other, out=None) -> Tensor
#' 
#' Computes :math:`\text{input} < \text{other}` element-wise.
#' 
#' The second argument can be a number or a tensor whose shape is
#' :ref:`broadcastable <broadcasting-semantics>` with the first argument.
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
#' take(input, index) -> Tensor
#' 
#' Returns a new tensor with the elements of :attr:`input` at the given indices.
#' The input tensor is treated as if it were viewed as a 1-D tensor. The result
#' takes the same shape as the indices.
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
#' index_select(input, dim, index, out=None) -> Tensor
#' 
#' Returns a new tensor which indexes the :attr:`input` tensor along dimension
#' :attr:`dim` using the entries in :attr:`index` which is a `LongTensor`.
#' 
#' The returned tensor has the same number of dimensions as the original tensor
#' (:attr:`input`).  The :attr:`dim`\ th dimension has the same size as the length
#' of :attr:`index`; other dimensions have the same size as in the original tensor.
#' 
#' .. note:: The returned tensor does **not** use the same storage as the original
#'           tensor.  If :attr:`out` has a different shape than expected, we
#'           silently change it to the correct shape, reallocating the underlying
#'           storage if necessary.
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
#' masked_select(input, mask, out=None) -> Tensor
#' 
#' Returns a new 1-D tensor which indexes the :attr:`input` tensor according to
#' the boolean mask :attr:`mask` which is a `BoolTensor`.
#' 
#' The shapes of the :attr:`mask` tensor and the :attr:`input` tensor don't need
#' to match, but they must be :ref:`broadcastable <broadcasting-semantics>`.
#' 
#' .. note:: The returned tensor does **not** use the same storage
#'           as the original tensor
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
#' nonzero(input, *, out=None, as_tuple=False) -> LongTensor or tuple of LongTensors
#' 
#' .. note::
#'     :func:`torch.nonzero(..., as_tuple=False) <torch.nonzero>` (default) returns a
#'     2-D tensor where each row is the index for a nonzero value.
#' 
#'     :func:`torch.nonzero(..., as_tuple=True) <torch.nonzero>` returns a tuple of 1-D
#'     index tensors, allowing for advanced indexing, so ``x[x.nonzero(as_tuple=True)]``
#'     gives all nonzero values of tensor ``x``. Of the returned tuple, each index tensor
#'     contains nonzero indices for a certain dimension.
#' 
#'     See below for more details on the two behaviors.
#' 
#' 
#' **When** :attr:`as_tuple` **is ``False`` (default)**:
#' 
#' Returns a tensor containing the indices of all non-zero elements of
#' :attr:`input`.  Each row in the result contains the indices of a non-zero
#' element in :attr:`input`. The result is sorted lexicographically, with
#' the last index changing the fastest (C-style).
#' 
#' If :attr:`input` has :math:`n` dimensions, then the resulting indices tensor
#' :attr:`out` is of size :math:`(z \times n)`, where :math:`z` is the total number of
#' non-zero elements in the :attr:`input` tensor.
#' 
#' **When** :attr:`as_tuple` **is ``True``**:
#' 
#' Returns a tuple of 1-D tensors, one for each dimension in :attr:`input`,
#' each containing the indices (in that dimension) of all non-zero elements of
#' :attr:`input` .
#' 
#' If :attr:`input` has :math:`n` dimensions, then the resulting tuple contains :math:`n`
#' tensors of size :math:`z`, where :math:`z` is the total number of
#' non-zero elements in the :attr:`input` tensor.
#' 
#' As a special case, when :attr:`input` has zero dimensions and a nonzero scalar
#' value, it is treated as a one-dimensional tensor with one element.
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
#' gather(input, dim, index, out=None, sparse_grad=False) -> Tensor
#' 
#' Gathers values along an axis specified by `dim`.
#' 
#' For a 3-D tensor the output is specified by::
#' 
#'     out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
#'     out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
#'     out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
#' 
#' If :attr:`input` is an n-dimensional tensor with size
#' :math:`(x_0, x_1..., x_{i-1}, x_i, x_{i+1}, ..., x_{n-1})`
#' and ``dim = i``, then :attr:`index` must be an :math:`n`-dimensional tensor with
#' size :math:`(x_0, x_1, ..., x_{i-1}, y, x_{i+1}, ..., x_{n-1})` where :math:`y \geq 1`
#' and :attr:`out` will have the same size as :attr:`index`.
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
#' addcmul(input, value=1, tensor1, tensor2, out=None) -> Tensor
#' 
#' Performs the element-wise multiplication of :attr:`tensor1`
#' by :attr:`tensor2`, multiply the result by the scalar :attr:`value`
#' and add it to :attr:`input`.
#' 
#' .. math::
#'     \text{out}_i = \text{input}_i + \text{value} \times \text{tensor1}_i \times \text{tensor2}_i
#' 
#' The shapes of :attr:`tensor`, :attr:`tensor1`, and :attr:`tensor2` must be
#' :ref:`broadcastable <broadcasting-semantics>`.
#' 
#' For inputs of type `FloatTensor` or `DoubleTensor`, :attr:`value` must be
#' a real number, otherwise an integer.
#'
#' @param input (Tensor) the tensor to be added
#' @param value (Number, optional) multiplier for :math:`tensor1 .* tensor2`
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
#' addcdiv(input, value=1, tensor1, tensor2, out=None) -> Tensor
#' 
#' Performs the element-wise division of :attr:`tensor1` by :attr:`tensor2`,
#' multiply the result by the scalar :attr:`value` and add it to :attr:`input`.
#' 
#' .. math::
#'     \text{out}_i = \text{input}_i + \text{value} \times \frac{\text{tensor1}_i}{\text{tensor2}_i}
#' 
#' The shapes of :attr:`input`, :attr:`tensor1`, and :attr:`tensor2` must be
#' :ref:`broadcastable <broadcasting-semantics>`.
#' 
#' For inputs of type `FloatTensor` or `DoubleTensor`, :attr:`value` must be
#' a real number, otherwise an integer.
#'
#' @param input (Tensor) the tensor to be added
#' @param value (Number, optional) multiplier for :math:`\text{tensor1} / \text{tensor2}`
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
#' lstsq(input, A, out=None) -> Tensor
#' 
#' Computes the solution to the least squares and least norm problems for a full
#' rank matrix :math:`A` of size :math:`(m \times n)` and a matrix :math:`B` of
#' size :math:`(m \times k)`.
#' 
#' If :math:`m \geq n`, :func:`lstsq` solves the least-squares problem:
#' 
#' .. math::
#' 
#'    \begin{array}{ll}
#'    \min_X & \|AX-B\|_2.
#'    \end{array}
#' 
#' If :math:`m < n`, :func:`lstsq` solves the least-norm problem:
#' 
#' .. math::
#' 
#'    \begin{array}{ll}
#'    \min_X & \|X\|_2 & \text{subject to} & AX = B.
#'    \end{array}
#' 
#' Returned tensor :math:`X` has shape :math:`(\max(m, n) \times k)`. The first :math:`n`
#' rows of :math:`X` contains the solution. If :math:`m \geq n`, the residual sum of squares
#' for the solution in each column is given by the sum of squares of elements in the
#' remaining :math:`m - n` rows of that column.
#' 
#' .. note::
#'     The case when :math:`m < n` is not supported on the GPU.
#'
#' @param input (Tensor) the matrix :math:`B`
#' @param A (Tensor) the :math:`m` by :math:`n` matrix :math:`A`
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
#' triangular_solve(input, A, upper=True, transpose=False, unitriangular=False) -> (Tensor, Tensor)
#' 
#' Solves a system of equations with a triangular coefficient matrix :math:`A`
#' and multiple right-hand sides :math:`b`.
#' 
#' In particular, solves :math:`AX = b` and assumes :math:`A` is upper-triangular
#' with the default keyword arguments.
#' 
#' `torch.triangular_solve(b, A)` can take in 2D inputs `b, A` or inputs that are
#' batches of 2D matrices. If the inputs are batches, then returns
#' batched outputs `X`
#'
#' @param input (Tensor) multiple right-hand sides of size :math:`(*, m, k)` where                :math:`*` is zero of more batch dimensions (:math:`b`)
#' @param A (Tensor) the input triangular coefficient matrix of size :math:`(*, m, m)`                where :math:`*` is zero or more batch dimensions
#' @param upper (bool, optional) whether to solve the upper-triangular system        of equations (default) or the lower-triangular system of equations. Default: ``True``.
#' @param transpose (bool, optional) whether :math:`A` should be transposed before        being sent into the solver. Default: ``False``.
#' @param unitriangular (bool, optional) whether :math:`A` is unit triangular.        If True, the diagonal elements of :math:`A` are assumed to be        1 and not referenced from :math:`A`. Default: ``False``.
#'
#' 
#'
#' @name torch_triangular_solve
#'
#' @export
NULL


#' Symeig
#'
#' symeig(input, eigenvectors=False, upper=True, out=None) -> (Tensor, Tensor)
#' 
#' This function returns eigenvalues and eigenvectors
#' of a real symmetric matrix :attr:`input` or a batch of real symmetric matrices,
#' represented by a namedtuple (eigenvalues, eigenvectors).
#' 
#' This function calculates all eigenvalues (and vectors) of :attr:`input`
#' such that :math:`\text{input} = V \text{diag}(e) V^T`.
#' 
#' The boolean argument :attr:`eigenvectors` defines computation of
#' both eigenvectors and eigenvalues or eigenvalues only.
#' 
#' If it is ``False``, only eigenvalues are computed. If it is ``True``,
#' both eigenvalues and eigenvectors are computed.
#' 
#' Since the input matrix :attr:`input` is supposed to be symmetric,
#' only the upper triangular portion is used by default.
#' 
#' If :attr:`upper` is ``False``, then lower triangular portion is used.
#' 
#' .. note:: The eigenvalues are returned in ascending order. If :attr:`input` is a batch of matrices,
#'           then the eigenvalues of each matrix in the batch is returned in ascending order.
#' 
#' .. note:: Irrespective of the original strides, the returned matrix `V` will
#'           be transposed, i.e. with strides `V.contiguous().transpose(-1, -2).stride()`.
#' 
#' .. note:: Extra care needs to be taken when backward through outputs. Such
#'           operation is really only stable when all eigenvalues are distinct.
#'           Otherwise, ``NaN`` can appear as the gradients are not properly defined.
#'
#' @param input (Tensor) the input tensor of size :math:`(*, n, n)` where `*` is zero or more                    batch dimensions consisting of symmetric matrices.
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
#' eig(input, eigenvectors=False, out=None) -> (Tensor, Tensor)
#' 
#' Computes the eigenvalues and eigenvectors of a real square matrix.
#' 
#' .. note::
#'     Since eigenvalues and eigenvectors might be complex, backward pass is supported only
#'     for :func:`torch.symeig`
#'
#' @param input (Tensor) the square matrix of shape :math:`(n \times n)` for which the eigenvalues and eigenvectors        will be computed
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
#' svd(input, some=True, compute_uv=True, out=None) -> (Tensor, Tensor, Tensor)
#' 
#' This function returns a namedtuple ``(U, S, V)`` which is the singular value
#' decomposition of a input real matrix or batches of real matrices :attr:`input` such that
#' :math:`input = U \times diag(S) \times V^T`.
#' 
#' If :attr:`some` is ``True`` (default), the method returns the reduced singular value decomposition
#' i.e., if the last two dimensions of :attr:`input` are ``m`` and ``n``, then the returned
#' `U` and `V` matrices will contain only :math:`min(n, m)` orthonormal columns.
#' 
#' If :attr:`compute_uv` is ``False``, the returned `U` and `V` matrices will be zero matrices
#' of shape :math:`(m \times m)` and :math:`(n \times n)` respectively. :attr:`some` will be ignored here.
#' 
#' .. note:: The singular values are returned in descending order. If :attr:`input` is a batch of matrices,
#'           then the singular values of each matrix in the batch is returned in descending order.
#' 
#' .. note:: The implementation of SVD on CPU uses the LAPACK routine `?gesdd` (a divide-and-conquer
#'           algorithm) instead of `?gesvd` for speed. Analogously, the SVD on GPU uses the MAGMA routine
#'           `gesdd` as well.
#' 
#' .. note:: Irrespective of the original strides, the returned matrix `U`
#'           will be transposed, i.e. with strides :code:`U.contiguous().transpose(-2, -1).stride()`
#' 
#' .. note:: Extra care needs to be taken when backward through `U` and `V`
#'           outputs. Such operation is really only stable when :attr:`input` is
#'           full rank with all distinct singular values. Otherwise, ``NaN`` can
#'           appear as the gradients are not properly defined. Also, notice that
#'           double backward will usually do an additional backward through `U` and
#'           `V` even if the original backward is only on `S`.
#' 
#' .. note:: When :attr:`some` = ``False``, the gradients on :code:`U[..., :, min(m, n):]`
#'           and :code:`V[..., :, min(m, n):]` will be ignored in backward as those vectors
#'           can be arbitrary bases of the subspaces.
#' 
#' .. note:: When :attr:`compute_uv` = ``False``, backward cannot be performed since `U` and `V`
#'           from the forward pass is required for the backward operation.
#'
#' @param input (Tensor) the input tensor of size :math:`(*, m, n)` where `*` is zero or more                    batch dimensions consisting of :math:`m \times n` matrices.
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
#' cholesky(input, upper=False, out=None) -> Tensor
#' 
#' Computes the Cholesky decomposition of a symmetric positive-definite
#' matrix :math:`A` or for batches of symmetric positive-definite matrices.
#' 
#' If :attr:`upper` is ``True``, the returned matrix ``U`` is upper-triangular, and
#' the decomposition has the form:
#' 
#' .. math::
#' 
#'   A = U^TU
#' 
#' If :attr:`upper` is ``False``, the returned matrix ``L`` is lower-triangular, and
#' the decomposition has the form:
#' 
#' .. math::
#' 
#'     A = LL^T
#' 
#' If :attr:`upper` is ``True``, and :math:`A` is a batch of symmetric positive-definite
#' matrices, then the returned tensor will be composed of upper-triangular Cholesky factors
#' of each of the individual matrices. Similarly, when :attr:`upper` is ``False``, the returned
#' tensor will be composed of lower-triangular Cholesky factors of each of the individual
#' matrices.
#'
#' @param input (Tensor) the input tensor :math:`A` of size :math:`(*, n, n)` where `*` is zero or more                batch dimensions consisting of symmetric positive-definite matrices.
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
#' cholesky_solve(input, input2, upper=False, out=None) -> Tensor
#' 
#' Solves a linear system of equations with a positive semidefinite
#' matrix to be inverted given its Cholesky factor matrix :math:`u`.
#' 
#' If :attr:`upper` is ``False``, :math:`u` is and lower triangular and `c` is
#' returned such that:
#' 
#' .. math::
#'     c = (u u^T)^{{-1}} b
#' 
#' If :attr:`upper` is ``True`` or not provided, :math:`u` is upper triangular
#' and `c` is returned such that:
#' 
#' .. math::
#'     c = (u^T u)^{{-1}} b
#' 
#' `torch.cholesky_solve(b, u)` can take in 2D inputs `b, u` or inputs that are
#' batches of 2D matrices. If the inputs are batches, then returns
#' batched outputs `c`
#'
#' @param input (Tensor) input matrix :math:`b` of size :math:`(*, m, k)`,                where :math:`*` is zero or more batch dimensions
#' @param input2 (Tensor) input matrix :math:`u` of size :math:`(*, m, m)`,                where :math:`*` is zero of more batch dimensions composed of                upper or lower triangular Cholesky factor
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
#' torch.solve(input, A, out=None) -> (Tensor, Tensor)
#' 
#' This function returns the solution to the system of linear
#' equations represented by :math:`AX = B` and the LU factorization of
#' A, in order as a namedtuple `solution, LU`.
#' 
#' `LU` contains `L` and `U` factors for LU factorization of `A`.
#' 
#' `torch.solve(B, A)` can take in 2D inputs `B, A` or inputs that are
#' batches of 2D matrices. If the inputs are batches, then returns
#' batched outputs `solution, LU`.
#' 
#' .. note::
#' 
#'     Irrespective of the original strides, the returned matrices
#'     `solution` and `LU` will be transposed, i.e. with strides like
#'     `B.contiguous().transpose(-1, -2).stride()` and
#'     `A.contiguous().transpose(-1, -2).stride()` respectively.
#'
#' @param input (Tensor) input matrix :math:`B` of size :math:`(*, m, k)` , where :math:`*`                is zero or more batch dimensions.
#' @param A (Tensor) input square matrix of size :math:`(*, m, m)`, where                :math:`*` is zero or more batch dimensions.
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
#' cholesky_inverse(input, upper=False, out=None) -> Tensor
#' 
#' Computes the inverse of a symmetric positive-definite matrix :math:`A` using its
#' Cholesky factor :math:`u`: returns matrix ``inv``. The inverse is computed using
#' LAPACK routines ``dpotri`` and ``spotri`` (and the corresponding MAGMA routines).
#' 
#' If :attr:`upper` is ``False``, :math:`u` is lower triangular
#' such that the returned tensor is
#' 
#' .. math::
#'     inv = (uu^{{T}})^{{-1}}
#' 
#' If :attr:`upper` is ``True`` or not provided, :math:`u` is upper
#' triangular such that the returned tensor is
#' 
#' .. math::
#'     inv = (u^T u)^{{-1}}
#'
#' @param input (Tensor) the input 2-D tensor :math:`u`, a upper or lower triangular           Cholesky factor
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
#' qr(input, some=True, out=None) -> (Tensor, Tensor)
#' 
#' Computes the QR decomposition of a matrix or a batch of matrices :attr:`input`,
#' and returns a namedtuple (Q, R) of tensors such that :math:`\text{input} = Q R`
#' with :math:`Q` being an orthogonal matrix or batch of orthogonal matrices and
#' :math:`R` being an upper triangular matrix or batch of upper triangular matrices.
#' 
#' If :attr:`some` is ``True``, then this function returns the thin (reduced) QR factorization.
#' Otherwise, if :attr:`some` is ``False``, this function returns the complete QR factorization.
#' 
#' .. note:: precision may be lost if the magnitudes of the elements of :attr:`input`
#'           are large
#' 
#' .. note:: While it should always give you a valid decomposition, it may not
#'           give you the same one across platforms - it will depend on your
#'           LAPACK implementation.
#'
#' @param input (Tensor) the input tensor of size :math:`(*, m, n)` where `*` is zero or more                batch dimensions consisting of matrices of dimension :math:`m \times n`.
#' @param some (bool, optional) Set to ``True`` for reduced QR decomposition and ``False`` for                complete QR decomposition.
#' @param out (tuple, optional) tuple of `Q` and `R` tensors                satisfying :code:`input = torch.matmul(Q, R)`.                The dimensions of `Q` and `R` are :math:`(*, m, k)` and :math:`(*, k, n)`                respectively, where :math:`k = \min(m, n)` if :attr:`some:` is ``True`` and                :math:`k = m` otherwise.
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
#' geqrf(input, out=None) -> (Tensor, Tensor)
#' 
#' This is a low-level function for calling LAPACK directly. This function
#' returns a namedtuple (a, tau) as defined in `LAPACK documentation for geqrf`_ .
#' 
#' You'll generally want to use :func:`torch.qr` instead.
#' 
#' Computes a QR decomposition of :attr:`input`, but without constructing
#' :math:`Q` and :math:`R` as explicit separate matrices.
#' 
#' Rather, this directly calls the underlying LAPACK function `?geqrf`
#' which produces a sequence of 'elementary reflectors'.
#' 
#' See `LAPACK documentation for geqrf`_ for further details.
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
#' orgqr(input, input2) -> Tensor
#' 
#' Computes the orthogonal matrix `Q` of a QR factorization, from the `(input, input2)`
#' tuple returned by :func:`torch.geqrf`.
#' 
#' This directly calls the underlying LAPACK function `?orgqr`.
#' See `LAPACK documentation for orgqr`_ for further details.
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
#' ormqr(input, input2, input3, left=True, transpose=False) -> Tensor
#' 
#' Multiplies `mat` (given by :attr:`input3`) by the orthogonal `Q` matrix of the QR factorization
#' formed by :func:`torch.geqrf` that is represented by `(a, tau)` (given by (:attr:`input`, :attr:`input2`)).
#' 
#' This directly calls the underlying LAPACK function `?ormqr`.
#' See `LAPACK documentation for ormqr`_ for further details.
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
#' lu_solve(input, LU_data, LU_pivots, out=None) -> Tensor
#' 
#' Returns the LU solve of the linear system :math:`Ax = b` using the partially pivoted
#' LU factorization of A from :meth:`torch.lu`.
#'
#' @param b (Tensor) the RHS tensor of size :math:`(*, m, k)`, where :math:`*`                is zero or more batch dimensions.
#' @param LU_data (Tensor) the pivoted LU factorization of A from :meth:`torch.lu` of size :math:`(*, m, m)`,                       where :math:`*` is zero or more batch dimensions.
#' @param LU_pivots (IntTensor) the pivots of the LU factorization from :meth:`torch.lu` of size :math:`(*, m)`,                           where :math:`*` is zero or more batch dimensions.                           The batch dimensions of :attr:`LU_pivots` must be equal to the batch dimensions of                           :attr:`LU_data`.
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
#' multinomial(input, num_samples, replacement=False, *, generator=None, out=None) -> LongTensor
#' 
#' Returns a tensor where each row contains :attr:`num_samples` indices sampled
#' from the multinomial probability distribution located in the corresponding row
#' of tensor :attr:`input`.
#' 
#' .. note::
#'     The rows of :attr:`input` do not need to sum to one (in which case we use
#'     the values as weights), but must be non-negative, finite and have
#'     a non-zero sum.
#' 
#' Indices are ordered from left to right according to when each was sampled
#' (first samples are placed in first column).
#' 
#' If :attr:`input` is a vector, :attr:`out` is a vector of size :attr:`num_samples`.
#' 
#' If :attr:`input` is a matrix with `m` rows, :attr:`out` is an matrix of shape
#' :math:`(m \times \text{num\_samples})`.
#' 
#' If replacement is ``True``, samples are drawn with replacement.
#' 
#' If not, they are drawn without replacement, which means that when a
#' sample index is drawn for a row, it cannot be drawn again for that row.
#' 
#' .. note::
#'     When drawn without replacement, :attr:`num_samples` must be lower than
#'     number of non-zero elements in :attr:`input` (or the min number of non-zero
#'     elements in each row of :attr:`input` if it is a matrix).
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
#' lgamma(input, out=None) -> Tensor
#' 
#' Computes the logarithm of the gamma function on :attr:`input`.
#' 
#' .. math::
#'     \text{out}_{i} = \log \Gamma(\text{input}_{i})
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
#' digamma(input, out=None) -> Tensor
#' 
#' Computes the logarithmic derivative of the gamma function on `input`.
#' 
#' .. math::
#'     \psi(x) = \frac{d}{dx} \ln\left(\Gamma\left(x\right)\right) = \frac{\Gamma'(x)}{\Gamma(x)}
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
#' polygamma(n, input, out=None) -> Tensor
#' 
#' Computes the :math:`n^{th}` derivative of the digamma function on :attr:`input`.
#' :math:`n \geq 0` is called the order of the polygamma function.
#' 
#' .. math::
#'     \psi^{(n)}(x) = \frac{d^{(n)}}{dx^{(n)}} \psi(x)
#' 
#' .. note::
#'     This function is not implemented for :math:`n \geq 2`.
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
#' erfinv(input, out=None) -> Tensor
#' 
#' Computes the inverse error function of each element of :attr:`input`.
#' The inverse error function is defined in the range :math:`(-1, 1)` as:
#' 
#' .. math::
#'     \mathrm{erfinv}(\mathrm{erf}(x)) = x
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
#' sign(input, out=None) -> Tensor
#' 
#' Returns a new tensor with the signs of the elements of :attr:`input`.
#' 
#' .. math::
#'     \text{out}_{i} = \operatorname{sgn}(\text{input}_{i})
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
#' dist(input, other, p=2) -> Tensor
#' 
#' Returns the p-norm of (:attr:`input` - :attr:`other`)
#' 
#' The shapes of :attr:`input` and :attr:`other` must be
#' :ref:`broadcastable <broadcasting-semantics>`.
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
#' atan2(input, other, out=None) -> Tensor
#' 
#' Element-wise arctangent of :math:`\text{input}_{i} / \text{other}_{i}`
#' with consideration of the quadrant. Returns a new tensor with the signed angles
#' in radians between vector :math:`(\text{other}_{i}, \text{input}_{i})`
#' and vector :math:`(1, 0)`. (Note that :math:`\text{other}_{i}`, the second
#' parameter, is the x-coordinate, while :math:`\text{input}_{i}`, the first
#' parameter, is the y-coordinate.)
#' 
#' The shapes of ``input`` and ``other`` must be
#' :ref:`broadcastable <broadcasting-semantics>`.
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
#' Does a linear interpolation of two tensors :attr:`start` (given by :attr:`input`) and :attr:`end` based
#' on a scalar or tensor :attr:`weight` and returns the resulting :attr:`out` tensor.
#' 
#' .. math::
#'     \text{out}_i = \text{start}_i + \text{weight}_i \times (\text{end}_i - \text{start}_i)
#' 
#' The shapes of :attr:`start` and :attr:`end` must be
#' :ref:`broadcastable <broadcasting-semantics>`. If :attr:`weight` is a tensor, then
#' the shapes of :attr:`weight`, :attr:`start`, and :attr:`end` must be :ref:`broadcastable <broadcasting-semantics>`.
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
#' histc(input, bins=100, min=0, max=0, out=None) -> Tensor
#' 
#' Computes the histogram of a tensor.
#' 
#' The elements are sorted into equal width bins between :attr:`min` and
#' :attr:`max`. If :attr:`min` and :attr:`max` are both zero, the minimum and
#' maximum values of the data are used.
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
#' fmod(input, other, out=None) -> Tensor
#' 
#' Computes the element-wise remainder of division.
#' 
#' The dividend and divisor may contain both for integer and floating point
#' numbers. The remainder has the same sign as the dividend :attr:`input`.
#' 
#' When :attr:`other` is a tensor, the shapes of :attr:`input` and
#' :attr:`other` must be :ref:`broadcastable <broadcasting-semantics>`.
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
#' remainder(input, other, out=None) -> Tensor
#' 
#' Computes the element-wise remainder of division.
#' 
#' The divisor and dividend may contain both for integer and floating point
#' numbers. The remainder has the same sign as the divisor.
#' 
#' When :attr:`other` is a tensor, the shapes of :attr:`input` and
#' :attr:`other` must be :ref:`broadcastable <broadcasting-semantics>`.
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
#' sort(input, dim=-1, descending=False, out=None) -> (Tensor, LongTensor)
#' 
#' Sorts the elements of the :attr:`input` tensor along a given dimension
#' in ascending order by value.
#' 
#' If :attr:`dim` is not given, the last dimension of the `input` is chosen.
#' 
#' If :attr:`descending` is ``True`` then the elements are sorted in descending
#' order by value.
#' 
#' A namedtuple of (values, indices) is returned, where the `values` are the
#' sorted values and `indices` are the indices of the elements in the original
#' `input` tensor.
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
#' argsort(input, dim=-1, descending=False, out=None) -> LongTensor
#' 
#' Returns the indices that sort a tensor along a given dimension in ascending
#' order by value.
#' 
#' This is the second value returned by :meth:`torch.sort`.  See its documentation
#' for the exact semantics of this method.
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
#' topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)
#' 
#' Returns the :attr:`k` largest elements of the given :attr:`input` tensor along
#' a given dimension.
#' 
#' If :attr:`dim` is not given, the last dimension of the `input` is chosen.
#' 
#' If :attr:`largest` is ``False`` then the `k` smallest elements are returned.
#' 
#' A namedtuple of `(values, indices)` is returned, where the `indices` are the indices
#' of the elements in the original `input` tensor.
#' 
#' The boolean option :attr:`sorted` if ``True``, will make sure that the returned
#' `k` elements are themselves sorted
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
#' renorm(input, p, dim, maxnorm, out=None) -> Tensor
#' 
#' Returns a tensor where each sub-tensor of :attr:`input` along dimension
#' :attr:`dim` is normalized such that the `p`-norm of the sub-tensor is lower
#' than the value :attr:`maxnorm`
#' 
#' .. note:: If the norm of a row is lower than `maxnorm`, the row is unchanged
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
#' equal(input, other) -> bool
#' 
#' ``True`` if two tensors have the same size and elements, ``False`` otherwise.
#' 
#' Example::
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
#' normal(mean, std, *, generator=None, out=None) -> Tensor
#' 
#' Returns a tensor of random numbers drawn from separate normal distributions
#' whose mean and standard deviation are given.
#' 
#' The :attr:`mean` is a tensor with the mean of
#' each output element's normal distribution
#' 
#' The :attr:`std` is a tensor with the standard deviation of
#' each output element's normal distribution
#' 
#' The shapes of :attr:`mean` and :attr:`std` don't need to match, but the
#' total number of elements in each tensor need to be the same.
#' 
#' .. note:: When the shapes do not match, the shape of :attr:`mean`
#'           is used as the shape for the returned output tensor
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
#' normal(mean=0.0, std, out=None) -> Tensor
#' 
#' Similar to the function above, but the means are shared among all drawn
#' elements.
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
#' normal(mean, std=1.0, out=None) -> Tensor
#' 
#' Similar to the function above, but the standard-deviations are shared among
#' all drawn elements.
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
#' normal(mean, std, size, *, out=None) -> Tensor
#' 
#' Similar to the function above, but the means and standard deviations are shared
#' among all drawn elements. The resulting tensor has size given by :attr:`size`.
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
#'     Arguments:
#'         tensor (Tensor): A tensor to check
#' 
#'     Returns:
#'         Tensor: ``A torch.Tensor with dtype torch.bool`` containing a True at each location of finite elements and False otherwise
#' 
#'     Example::
#' 
#'         >>> torch.isfinite(torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]))
#'         tensor([True,  False,  True,  False,  False])
#'

#'
#' 
#'
#' @name torch_isfinite
#'
#' @export
NULL
