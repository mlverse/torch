# -> abs: e5dcc7c79de65d8dfd1b04a5b19003df <-
#'
#' @name torch_abs
#'
#' @examples
#'
#' torch_abs(torch_tensor(c(-1, -2, 3)))
NULL

# -> angle: 9621fdde4144667b5f3fe4c9f7edc853 <-
#'
#' @name torch_angle
#'
#' @examples
#'
#' torch_angle(torch_tensor(c(-1 + 1i, -2 + 2i, 3 - 3i)))*180/3.14159
NULL

# -> real: ce82bd172d0842e27c33861733d35928 <-
#'
#' @name torch_real
#'
#' @examples
#'
#' torch_real(torch_tensor(c(-1 + 1i, -2 + 2i, 3 - 3i)))
NULL

# -> imag: 57addec87c3caad9ba6674074a165f7c <-
#'
#' @name torch_imag
#'
#' @examples
#'
#' torch_imag(torch_tensor(c(-1 + 1i, -2 + 2i, 3 - 3i)))
NULL

# -> conj: a4675ce91c7382434d5f94e70d9da092 <-
#'
#' @name torch_conj
#'
#' @examples
#'
#' torch_conj(torch_tensor(c(-1 + 1i, -2 + 2i, 3 - 3i)))
NULL

# -> acos: de947e3c803bf4e24ab85d718ffecbbe <-
#'
#' @name torch_acos
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_acos(a)
NULL

# -> avg_pool1d: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_avg_pool1d
#'
#' @examples
#'
#' 
NULL

# -> adaptive_avg_pool1d: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_adaptive_avg_pool1d
#'
#' @examples
#'
#' 
NULL

# -> add: e5234c5c58dc7d552ac91d3ae1586ebe <-
#'
#' @name torch_add
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_add(a, 20)
#'
#'
#' a = torch_randn(c(4))
#' a
#' b = torch_randn(c(4, 1))
#' b
#' torch_add(a, 10, b)
NULL

# -> addmv: 95a7ca8251046a178e094a9b3b868fcc <-
#'
#' @name torch_addmv
#'
#' @examples
#'
#' M = torch_randn(c(2))
#' mat = torch_randn(c(2, 3))
#' vec = torch_randn(c(3))
#' torch_addmv(M, mat, vec)
NULL

# -> addr: 7fe5b137e75dbe0559b35e9bb822b5bc <-
#'
#' @name torch_addr
#'
#' @examples
#'
#' vec1 = torch_arange(1., 4.)
#' vec2 = torch_arange(1., 3.)
#' M = torch_zeros(c(3, 2))
#' torch_addr(M, vec1, vec2)
NULL

# -> allclose: 8c3297713c8bf069c062982c901b6cba <-
#'
#' @name torch_allclose
#'
#' @examples
#'
#' torch_allclose(torch_tensor(c(10000., 1e-07)), torch_tensor(c(10000.1, 1e-08)))
#' torch_allclose(torch_tensor(c(10000., 1e-08)), torch_tensor(c(10000.1, 1e-09)))
#' torch_allclose(torch_tensor([1.0, float('nan')]), torch_tensor([1.0, float('nan')]))
#' torch_allclose(torch_tensor([1.0, float('nan')]), torch_tensor([1.0, float('nan')]), equal_nan=TRUE)
NULL

# -> arange: 3b9e05598c28f78d92ed1f2445214942 <-
#'
#' @name torch_arange
#'
#' @examples
#'
#' torch_arange(5)
#' torch_arange(1, 4)
#' torch_arange(1, 2.5, 0.5)
NULL

# -> argmax: d282510a6dcfc0e72f3200971137777d <-
#'
#' @name torch_argmax
#'
#' @examples
#'
#' a = torch_randn(c(4, 4))
#' a
#' torch_argmax(a)
#'
#'
#' a = torch_randn(c(4, 4))
#' a
#' torch_argmax(a, dim=1)
NULL

# -> argmin: 4803748c67670143bd6efdfb452e5a7e <-
#'
#' @name torch_argmin
#'
#' @examples
#'
#' a = torch_randn(c(4, 4))
#' a
#' torch_argmin(a)
#'
#'
#' a = torch_randn(c(4, 4))
#' a
#' torch_argmin(a, dim=1)
NULL

# -> as_strided: 01d0400373250adc462d043cf0eab2a3 <-
#'
#' @name torch_as_strided
#'
#' @examples
#'
#' x = torch_randn(c(3, 3))
#' x
#' t = torch_as_strided(x, list(2, 2), list(1, 2))
#' t
#' t = torch_as_strided(x, list(2, 2), list(1, 2), 1)
NULL

# -> asin: 06862ae1b297aaab8a06dc17014cc68d <-
#'
#' @name torch_asin
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_asin(a)
NULL

# -> atan: deb98403fe30b6df0380ed4ae9387292 <-
#'
#' @name torch_atan
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_atan(a)
NULL

# -> baddbmm: 6dcb499552161b4c566226d79cb2784d <-
#'
#' @name torch_baddbmm
#'
#' @examples
#'
#' M = torch_randn(c(10, 3, 5))
#' batch1 = torch_randn(c(10, 3, 4))
#' batch2 = torch_randn(c(10, 4, 5))
#' torch_baddbmm(M, batch1, batch2)$size()
NULL

# -> bartlett_window: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_bartlett_window
#'
#' @examples
#'
#' 
NULL

# -> bernoulli: 9b88ad8f9841b4f4554546b7e0810cc8 <-
#'
#' @name torch_bernoulli
#'
#' @examples
#'
#' a = torch_empty(3, 3)$uniform_list(0, 1)  # generate a uniform random matrix with range c(0, 1)
#' a
#' torch_bernoulli(a)
#' a = torch_ones(c(3, 3)) # probability of drawing "1" is 1
#' torch_bernoulli(a)
#' a = torch_zeros(c(3, 3)) # probability of drawing "1" is 0
#' torch_bernoulli(a)
NULL

# -> bincount: 8f376efeb3f93eee77fa02e8daaf71a6 <-
#'
#' @name torch_bincount
#'
#' @examples
#'
#' input = torch_randint(0, 8, list(5,), dtype=torch_int64())
#' weights = torch_linspace(0, 1, steps=5)
#' input, weights
#' torch_bincount(input)
#' input$bincount(weights)
NULL

# -> bitwise_not: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_bitwise_not
#'
#' @examples
#'
#' 
NULL

# -> logical_not: 7bc862f74fd8849a25f5043c76d2be75 <-
#'
#' @name torch_logical_not
#'
#' @examples
#'
#' torch_logical_not(torch_tensor(c(TRUE, FALSE)))
#' torch_logical_not(torch_tensor(c(0, 1, -10), dtype=torch_int8()))
#' torch_logical_not(torch_tensor(c(0., 1.5, -10.), dtype=torch_double()))
#' torch_logical_not(torch_tensor(c(0., 1., -10.), dtype=torch_double()), out=torch_empty(3, dtype=torch_int16()))
NULL

# -> logical_xor: f160bb410361a340b5e68613a4d89c49 <-
#'
#' @name torch_logical_xor
#'
#' @examples
#'
#' torch_logical_xor(torch_tensor(c(TRUE, FALSE, TRUE)), torch_tensor(c(TRUE, FALSE, FALSE)))
#' a = torch_tensor(c(0, 1, 10, 0), dtype=torch_int8())
#' b = torch_tensor(c(4, 0, 1, 0), dtype=torch_int8())
#' torch_logical_xor(a, b)
#' torch_logical_xor(a$double(), b$double())
#' torch_logical_xor(a$double(), b)
#' torch_logical_xor(a, b, out=torch_empty(4, dtype=torch_bool()))
NULL

# -> blackman_window: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_blackman_window
#'
#' @examples
#'
#' 
NULL

# -> bmm: 94a2609444054e49db409a2c41e8e395 <-
#'
#' @name torch_bmm
#'
#' @examples
#'
#' input = torch_randn(c(10, 3, 4))
#' mat2 = torch_randn(c(10, 4, 5))
#' res = torch_bmm(input, mat2)
#' res$size()
NULL

# -> broadcast_tensors: 40a0612dc7ad6fe698efc663dee38fca <-
#'
#' @name torch_broadcast_tensors
#'
#' @examples
#'
#' x = torch_arange(3)$view(1, 3)
#' y = torch_arange(2)$view(2, 1)
#' a, b = torch_broadcast_tensors(x, y)
#' a$size()
#' a
NULL

# -> cat: f59111e8a11eeec3514d7e89b8c9df38 <-
#'
#' @name torch_cat
#'
#' @examples
#'
#' x = torch_randn(c(2, 3))
#' x
#' torch_cat(list(x, x, x), 0)
#' torch_cat(list(x, x, x), 1)
NULL

# -> ceil: 239c07102107a5f1bee737a4c4dda5cb <-
#'
#' @name torch_ceil
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_ceil(a)
NULL

# -> chain_matmul: d5fdd723ee4defcebfcda3f59c2d7583 <-
#'
#' @name torch_chain_matmul
#'
#' @examples
#'
#' a = torch_randn(c(3, 4))
#' b = torch_randn(c(4, 5))
#' c = torch_randn(c(5, 6))
#' d = torch_randn(c(6, 7))
#' torch_chain_matmul(a, b, c, d)
NULL

# -> chunk: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_chunk
#'
#' @examples
#'
#' 
NULL

# -> clamp: e0edd0190272b6f2d6da83e5a655a576 <-
#'
#' @name torch_clamp
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_clamp(a, min=-0.5, max=0.5)
#'
#'
#' a = torch_randn(c(4))
#' a
#' torch_clamp(a, min=0.5)
#'
#'
#' a = torch_randn(c(4))
#' a
#' torch_clamp(a, max=0.5)
NULL

# -> conv1d: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_conv1d
#'
#' @examples
#'
#' 
NULL

# -> conv2d: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_conv2d
#'
#' @examples
#'
#' 
NULL

# -> conv3d: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_conv3d
#'
#' @examples
#'
#' 
NULL

# -> conv_tbc: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_conv_tbc
#'
#' @examples
#'
#' 
NULL

# -> conv_transpose1d: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_conv_transpose1d
#'
#' @examples
#'
#' 
NULL

# -> conv_transpose2d: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_conv_transpose2d
#'
#' @examples
#'
#' 
NULL

# -> conv_transpose3d: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_conv_transpose3d
#'
#' @examples
#'
#' 
NULL

# -> cos: 5192040f1da2ab0b8d6b0bf3e9350bff <-
#'
#' @name torch_cos
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_cos(a)
NULL

# -> cosh: e5f66f0ceaa03000e5d9758a15720e11 <-
#'
#' @name torch_cosh
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_cosh(a)
NULL

# -> cumsum: bfe9884a4685aef797f8b6217cb43134 <-
#'
#' @name torch_cumsum
#'
#' @examples
#'
#' a = torch_randn(c(10))
#' a
#' torch_cumsum(a, dim=0)
NULL

# -> cumprod: c2bd9e3600eca4d6b76296ec16b33c8a <-
#'
#' @name torch_cumprod
#'
#' @examples
#'
#' a = torch_randn(c(10))
#' a
#' torch_cumprod(a, dim=0)
#' ac(5) = 0.0
#' torch_cumprod(a, dim=0)
NULL

# -> det: 52d605bf0ed56dbe5250fe877a58cf67 <-
#'
#' @name torch_det
#'
#' @examples
#'
#' A = torch_randn(c(3, 3))
#' torch_det(A)
#' A = torch_randn(c(3, 2, 2))
#' A
#' A$det()
NULL

# -> diag_embed: 5d05dd914f0c95d15c53edcb24b102b9 <-
#'
#' @name torch_diag_embed
#'
#' @examples
#'
#' a = torch_randn(c(2, 3))
#' torch_diag_embed(a)
#' torch_diag_embed(a, offset=1, dim1=0, dim2=2)
NULL

# -> diagflat: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_diagflat
#'
#' @examples
#'
#' 
NULL

# -> diagonal: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_diagonal
#'
#' @examples
#'
#' 
NULL

# -> div: 00a982e99002adfd039a1a8392aabc5a <-
#'
#' @name torch_div
#'
#' @examples
#'
#' a = torch_randn(c(5))
#' a
#' torch_div(a, 0.5)
#'
#'
#' a = torch_randn(c(4, 4))
#' a
#' b = torch_randn(c(4))
#' b
#' torch_div(a, b)
NULL

# -> dot: f9059f8763a0a2d50613c46c1b74d861 <-
#'
#' @name torch_dot
#'
#' @examples
#'
#' torch_dot(torch_tensor(c(2, 3)), torch_tensor(c(2, 1)))
NULL

# -> einsum: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_einsum
#'
#' @examples
#'
#' 
NULL

# -> empty: 0962765c175f729cb8042eed65dc3ee3 <-
#'
#' @name torch_empty
#'
#' @examples
#'
#' torch_empty(2, 3)
NULL

# -> empty_like: 377c54c571c47d42e428b19201764263 <-
#'
#' @name torch_empty_like
#'
#' @examples
#'
#' torch_empty(list(2,3), dtype=torch_int64())
NULL

# -> empty_strided: 46b420ae46859eccc00bc9ffd70bd09b <-
#'
#' @name torch_empty_strided
#'
#' @examples
#'
#' a = torch_empty_strided(list(2, 3), list(1, 2))
#' a
#' a$stride()
#' a$size()
NULL

# -> erf: 6df267ff8119f54fd357a05e105f5168 <-
#'
#' @name torch_erf
#'
#' @examples
#'
#' torch_erf(torch_tensor(c(0, -1., 10.)))
NULL

# -> erfc: c133a5a098565b1346023122c5945802 <-
#'
#' @name torch_erfc
#'
#' @examples
#'
#' torch_erfc(torch_tensor(c(0, -1., 10.)))
NULL

# -> exp: 1fe4878ed948c1efe4ad1fb0b87ad82d <-
#'
#' @name torch_exp
#'
#' @examples
#'
#' torch_exp(torch_tensor([0, math$log(2.)]))
NULL

# -> expm1: fceee3efca02cabb971ad297660e1087 <-
#'
#' @name torch_expm1
#'
#' @examples
#'
#' torch_expm1list(torch_tensor([0, math$log(2.)]))
NULL

# -> eye: 2fe5a458ffa2244d8fdbb04a0f9f23b0 <-
#'
#' @name torch_eye
#'
#' @examples
#'
#' torch_eye(3)
NULL

# -> flatten: 68cb4abaf3139c0692565056e8a2c8d1 <-
#'
#' @name torch_flatten
#'
#' @examples
#'
#' t = torch_tensor(c([[1, 2),
#' torch_flatten(t)
#' torch_flatten(t, start_dim=1)
NULL

# -> floor: 3c997f6f3c41725e799823c9e8c924d2 <-
#'
#' @name torch_floor
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_floor(a)
NULL

# -> frac: 0fd8f6b68e156eea7946996ac8c5535f <-
#'
#' @name torch_frac
#'
#' @examples
#'
#' torch_frac(torch_tensor(c(1, 2.5, -3.2)))
NULL

# -> full: 3829335d5969b2928c3abf7c74a2c32e <-
#'
#' @name torch_full
#'
#' @examples
#'
#' torch_full(list(2, 3), 3.141592)
NULL

# -> full_like: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_full_like
#'
#' @examples
#'
#' 
NULL

# -> hann_window: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_hann_window
#'
#' @examples
#'
#' 
NULL

# -> hamming_window: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_hamming_window
#'
#' @examples
#'
#' 
NULL

# -> ger: bdb4f502392d061b8224371f41502ea1 <-
#'
#' @name torch_ger
#'
#' @examples
#'
#' v1 = torch_arange(1., 5.)
#' v2 = torch_arange(1., 4.)
#' torch_ger(v1, v2)
NULL

# -> fft: 14a19be0e41f8e8eaa49c1055e95aaf4 <-
#'
#' @name torch_fft
#'
#' @examples
#'
#' # unbatched 2D FFT
#' x = torch_randn(c(4, 3, 2))
#' torch_fft(x, 2)
#' # batched 1D FFT
#' torch_fft(x, 1)
#' # arbitrary number of batch dimensions, 2D FFT
#' x = torch_randn(c(3, 3, 5, 5, 2))
#' y = torch_fft(x, 2)
#' y$shape
NULL

# -> ifft: 9e679f9ae3ea638ed2f7dc45b69b63b6 <-
#'
#' @name torch_ifft
#'
#' @examples
#'
#' x = torch_randn(c(3, 3, 2))
#' x
#' y = torch_fft(x, 2)
#' torch_ifft(y, 2)  # recover x
NULL

# -> rfft: 8a2b61d4b0d12db349aaa5c7994ffca4 <-
#'
#' @name torch_rfft
#'
#' @examples
#'
#' x = torch_randn(c(5, 5))
#' torch_rfft(x, 2)$shape
#' torch_rfft(x, 2, onesided=FALSE)$shape
NULL

# -> irfft: 78afad7a54b15a6cdd6a32220ce39068 <-
#'
#' @name torch_irfft
#'
#' @examples
#'
#' x = torch_randn(c(4, 4))
#' torch_rfft(x, 2, onesided=TRUE)$shape
#'     >>>
#' # notice that with onesided=TRUE, output size does not determine the original signal size
#' x = torch_randn(c(4, 5))
#' torch_rfft(x, 2, onesided=TRUE)$shape
#'     >>>
#' # now we use the original shape to recover x
#' x
#' y = torch_rfft(x, 2, onesided=TRUE)
#' torch_irfft(y, 2, onesided=TRUE, signal_sizes=x$shape)  # recover x
NULL

# -> inverse: e6453516271a376f1c60043ddb21f01b <-
#'
#' @name torch_inverse
#'
#' @examples
#'
#' x = torch_rand(4, 4)
#' y = torch_inverse(x)
#' z = torch_mm(x, y)
#' z
#' torch_max(torch_abs(z - torch_eye(4))) # Max non-zero
#' # Batched inverse example
#' x = torch_randn(c(2, 3, 4, 4))
#' y = torch_inverse(x)
#' z = torch_matmul(x, y)
#' torch_max(torch_abs(z - torch_eye(4)$expand_as(x))) # Max non-zero
NULL

# -> isnan: 5bbd181d5054fa8c56ec44787620b0dd <-
#'
#' @name torch_isnan
#'
#' @examples
#'
#' torch_isnan(torch_tensor([1, float('nan'), 2]))
NULL

# -> is_floating_point: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_is_floating_point
#'
#' @examples
#'
#' 
NULL

# -> kthvalue: 14ef5044026a5d4768363e051173efa9 <-
#'
#' @name torch_kthvalue
#'
#' @examples
#'
#' x = torch_arange(1., 6.)
#' x
#' torch_kthvalue(x, 4)
#' x=torch_arange(1.,7.)$resize_list(2,3)
#' x
#' torch_kthvalue(x, 2, 0, TRUE)
NULL

# -> linspace: 7ed06dd488b891d514eb151681a86b41 <-
#'
#' @name torch_linspace
#'
#' @examples
#'
#' torch_linspace(3, 10, steps=5)
#' torch_linspace(-10, 10, steps=5)
#' torch_linspace(start=-10, end=10, steps=5)
#' torch_linspace(start=-10, end=10, steps=1)
NULL

# -> log: 878d7cd15156b95b8ed35db633a1fdf1 <-
#'
#' @name torch_log
#'
#' @examples
#'
#' a = torch_randn(c(5))
#' a
#' torch_log(a)
NULL

# -> log10: d00371ce4dc5e2fb821590110de44d9e <-
#'
#' @name torch_log10
#'
#' @examples
#'
#' a = torch_rand(5)
#' a
#' torch_log10list(a)
NULL

# -> log1p: 04b60fd63ecc4b6cec219f9d6208009c <-
#'
#' @name torch_log1p
#'
#' @examples
#'
#' a = torch_randn(c(5))
#' a
#' torch_log1p(a)
NULL

# -> log2: c3e8d550a6bef1520ceaba718121518e <-
#'
#' @name torch_log2
#'
#' @examples
#'
#' a = torch_rand(5)
#' a
#' torch_log2list(a)
NULL

# -> logdet: 175ddcc25def953620c12c230860c16f <-
#'
#' @name torch_logdet
#'
#' @examples
#'
#' A = torch_randn(c(3, 3))
#' torch_det(A)
#' torch_logdet(A)
#' A
#' A$det()
#' A$det()$log()
NULL

# -> logspace: e230aee1021ed098cdda804a561048e0 <-
#'
#' @name torch_logspace
#'
#' @examples
#'
#' torch_logspace(start=-10, end=10, steps=5)
#' torch_logspace(start=0.1, end=1.0, steps=5)
#' torch_logspace(start=0.1, end=1.0, steps=1)
#' torch_logspace(start=2, end=2, steps=1, base=2)
NULL

# -> logsumexp: adbac23c0616322708602e11d5210d9c <-
#'
#' @name torch_logsumexp
#'
#' @examples
#'
#' a = torch_randn(c(3, 3))
#' torch_logsumexp(a, 1)
NULL

# -> matmul: 7de3f09d19269840c9448acaaad91714 <-
#'
#' @name torch_matmul
#'
#' @examples
#'
#' # vector x vector
#' tensor1 = torch_randn(c(3))
#' tensor2 = torch_randn(c(3))
#' torch_matmul(tensor1, tensor2)$size()
#' # matrix x vector
#' tensor1 = torch_randn(c(3, 4))
#' tensor2 = torch_randn(c(4))
#' torch_matmul(tensor1, tensor2)$size()
#' # batched matrix x broadcasted vector
#' tensor1 = torch_randn(c(10, 3, 4))
#' tensor2 = torch_randn(c(4))
#' torch_matmul(tensor1, tensor2)$size()
#' # batched matrix x batched matrix
#' tensor1 = torch_randn(c(10, 3, 4))
#' tensor2 = torch_randn(c(10, 4, 5))
#' torch_matmul(tensor1, tensor2)$size()
#' # batched matrix x broadcasted matrix
#' tensor1 = torch_randn(c(10, 3, 4))
#' tensor2 = torch_randn(c(4, 5))
#' torch_matmul(tensor1, tensor2)$size()
NULL

# -> matrix_rank: 12524e09a5466173601049872f25a568 <-
#'
#' @name torch_matrix_rank
#'
#' @examples
#'
#' a = torch_eye(10)
#' torch_matrix_rank(a)
#' b = torch_eye(10)
#' bc(0, 0) = 0
#' torch_matrix_rank(b)
NULL

# -> matrix_power: bec29ab52f4ad873109be6b2aeea068d <-
#'
#' @name torch_matrix_power
#'
#' @examples
#'
#' a = torch_randn(c(2, 2, 2))
#' a
#' torch_matrix_power(a, 3)
NULL

# -> max: 129d62f124e5bfd94fcda2a8008e2689 <-
#'
#' @name torch_max
#'
#' @examples
#'
#' a = torch_randn(c(1, 3))
#' a
#' torch_max(a)
#'
#'
#' a = torch_randn(c(4, 4))
#' a
#' torch_max(a, 1)
#'
#'
#' a = torch_randn(c(4))
#' a
#' b = torch_randn(c(4))
#' b
#' torch_max(a, b)
NULL

# -> mean: 5660b20f12e3b95de11939e779e542a2 <-
#'
#' @name torch_mean
#'
#' @examples
#'
#' a = torch_randn(c(1, 3))
#' a
#' torch_mean(a)
#'
#'
#' a = torch_randn(c(4, 4))
#' a
#' torch_mean(a, 1)
#' torch_mean(a, 1, TRUE)
NULL

# -> median: bf4e2b607f0873ec10469b01f35c5937 <-
#'
#' @name torch_median
#'
#' @examples
#'
#' a = torch_randn(c(1, 3))
#' a
#' torch_median(a)
#'
#'
#' a = torch_randn(c(4, 5))
#' a
#' torch_median(a, 1)
NULL

# -> min: 1d3170de29db8837306466047a756ce7 <-
#'
#' @name torch_min
#'
#' @examples
#'
#' a = torch_randn(c(1, 3))
#' a
#' torch_min(a)
#'
#'
#' a = torch_randn(c(4, 4))
#' a
#' torch_min(a, 1)
#'
#'
#' a = torch_randn(c(4))
#' a
#' b = torch_randn(c(4))
#' b
#' torch_min(a, b)
NULL

# -> mm: 31d22540d159e4ebc7d64db1125f75b7 <-
#'
#' @name torch_mm
#'
#' @examples
#'
#' mat1 = torch_randn(c(2, 3))
#' mat2 = torch_randn(c(3, 3))
#' torch_mm(mat1, mat2)
NULL

# -> mode: 790faaee05b786fcbfbbad8766c8e125 <-
#'
#' @name torch_mode
#'
#' @examples
#'
#' a = torch_randint(10, list(5,))
#' a
#' b = a + list(torch_randn(c(50, 1)) * 5)$long()
#' torch_mode(b, 0)
NULL

# -> mul: db651f9e0e3787499cdfefa03bf0027e <-
#'
#' @name torch_mul
#'
#' @examples
#'
#' a = torch_randn(c(3))
#' a
#' torch_mul(a, 100)
#'
#'
#' a = torch_randn(c(4, 1))
#' a
#' b = torch_randn(c(1, 4))
#' b
#' torch_mul(a, b)
NULL

# -> mv: 88257ae2d0a34bd3285a88807a5b20cc <-
#'
#' @name torch_mv
#'
#' @examples
#'
#' mat = torch_randn(c(2, 3))
#' vec = torch_randn(c(3))
#' torch_mv(mat, vec)
NULL

# -> mvlgamma: a06bdd1ab524cd8b528bd5b6d85bb9e2 <-
#'
#' @name torch_mvlgamma
#'
#' @examples
#'
#' a = torch_empty(2, 3)$uniform_list(1, 2)
#' a
#' torch_mvlgamma(a, 2)
NULL

# -> narrow: e4e974f281419fabf23a337f1d28c4b2 <-
#'
#' @name torch_narrow
#'
#' @examples
#'
#' x = torch_tensor(c([1, 2, 3], [4, 5, 6], [7, 8, 9]))
#' torch_narrow(x, 0, 0, 2)
#' torch_narrow(x, 1, 1, 2)
NULL

# -> ones: f1494db89d4d1b821b559a6d55f571bf <-
#'
#' @name torch_ones
#'
#' @examples
#'
#' torch_ones(c(2, 3))
#' torch_ones(c(5))
NULL

# -> ones_like: d8cad9e7e1a614325aeccaed736f99d4 <-
#'
#' @name torch_ones_like
#'
#' @examples
#'
#' input = torch_empty(2, 3)
#' torch_ones_like(input)
NULL

# -> cdist: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_cdist
#'
#' @examples
#'
#' 
NULL

# -> pdist: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_pdist
#'
#' @examples
#'
#' 
NULL

# -> cosine_similarity: b9752bed49729683d719eb23caa4328c <-
#'
#' @name torch_cosine_similarity
#'
#' @examples
#'
#' input1 = torch_randn(c(100, 128))
#' input2 = torch_randn(c(100, 128))
#' output = F$cosine_similarity(input1, input2)
#' print(output)
NULL

# -> pixel_shuffle: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_pixel_shuffle
#'
#' @examples
#'
#' 
NULL

# -> pinverse: 9278cbb5b1cc0259f2e7c25d94d9ee37 <-
#'
#' @name torch_pinverse
#'
#' @examples
#'
#' input = torch_randn(c(3, 5))
#' input
#' torch_pinverse(input)
#' # Batched pinverse example
#' a = torch_randn(c(2,6,3))
#' b = torch_pinverse(a)
#' torch_matmul(b, a)
NULL

# -> rand: e68d9f262ad9e5c98381551177852f0a <-
#'
#' @name torch_rand
#'
#' @examples
#'
#' torch_rand(4)
#' torch_rand(2, 3)
NULL

# -> rand_like: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_rand_like
#'
#' @examples
#'
#' 
NULL

# -> randint: 26b40bb65dea39c5661a2110b4fd7198 <-
#'
#' @name torch_randint
#'
#' @examples
#'
#' torch_randint(3, 5, list(3,))
#' torch_randint(10, list(2, 2))
#' torch_randint(3, 10, list(2, 2))
NULL

# -> randint_like: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_randint_like
#'
#' @examples
#'
#' 
NULL

# -> randn: 207e9f15e08ccfb883eef97838abbf0d <-
#'
#' @name torch_randn
#'
#' @examples
#'
#' torch_randn(c(4))
#' torch_randn(c(2, 3))
NULL

# -> randn_like: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_randn_like
#'
#' @examples
#'
#' 
NULL

# -> randperm: cb2f85bc9dce124f3a32d17e9c0e0a16 <-
#'
#' @name torch_randperm
#'
#' @examples
#'
#' torch_randperm(4)
NULL

# -> range: 0e7e7a272ee7d24e247a5ce9db1ce606 <-
#'
#' @name torch_range
#'
#' @examples
#'
#' torch_range(1, 4)
#' torch_range(1, 4, 0.5)
NULL

# -> reciprocal: b411de0d1e22b20b4da81226ebe2b16d <-
#'
#' @name torch_reciprocal
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_reciprocal(a)
NULL

# -> neg: df492a40744b54fc918863e0cdc224df <-
#'
#' @name torch_neg
#'
#' @examples
#'
#' a = torch_randn(c(5))
#' a
#' torch_neg(a)
NULL

# -> repeat_interleave: 98b685aa5bac07a476355bff55dc32e6 <-
#'
#' @name torch_repeat_interleave
#'
#' @examples
#'
#' x = torch_tensor(c(1, 2, 3))
#' x$repeat_interleave(2)
#' y = torch_tensor(c([1, 2], [3, 4]))
#' torch_repeat_interleave(y, 2)
#' torch_repeat_interleave(y, 3, dim=1)
#' torch_repeat_interleave(y, torch_tensor(c(1, 2)), dim=0)
#'
#'
#' 
NULL

# -> reshape: f3082fb6e75532d1001838e1878c77a1 <-
#'
#' @name torch_reshape
#'
#' @examples
#'
#' a = torch_arange(4.)
#' torch_reshape(a, list(2, 2))
#' b = torch_tensor(c([0, 1], [2, 3]))
#' torch_reshape(b, list(-1,))
NULL

# -> round: 8645352fbb37af2e4d8a88df678d7cf6 <-
#'
#' @name torch_round
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_round(a)
NULL

# -> rrelu_: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_rrelu_
#'
#' @examples
#'
#' 
NULL

# -> relu_: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_relu_
#'
#' @examples
#'
#' 
NULL

# -> rsqrt: f16013bcf94efc81744b610e39f56f03 <-
#'
#' @name torch_rsqrt
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_rsqrt(a)
NULL

# -> selu_: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_selu_
#'
#' @examples
#'
#' 
NULL

# -> celu_: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_celu_
#'
#' @examples
#'
#' 
NULL

# -> sigmoid: f1e83a8077fa44fdbe8e58b54b3c1a0c <-
#'
#' @name torch_sigmoid
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_sigmoid(a)
NULL

# -> sin: 96b72203e615e78f31c1e3131b033f6e <-
#'
#' @name torch_sin
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_sin(a)
NULL

# -> sinh: 1954df08b82ff1b688e9ea9b322337a6 <-
#'
#' @name torch_sinh
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_sinh(a)
NULL

# -> slogdet: 9c0ac4d84a56d68e9b0a6959e3938a39 <-
#'
#' @name torch_slogdet
#'
#' @examples
#'
#' A = torch_randn(c(3, 3))
#' A
#' torch_det(A)
#' torch_logdet(A)
#' torch_slogdet(A)
NULL

# -> split: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_split
#'
#' @examples
#'
#' 
NULL

# -> squeeze: 4e20748b9e924bbc20cf37321ad7c149 <-
#'
#' @name torch_squeeze
#'
#' @examples
#'
#' x = torch_zeros(c(2, 1, 2, 1, 2))
#' x$size()
#' y = torch_squeeze(x)
#' y$size()
#' y = torch_squeeze(x, 0)
#' y$size()
#' y = torch_squeeze(x, 1)
#' y$size()
NULL

# -> stack: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_stack
#'
#' @examples
#'
#' 
NULL

# -> stft: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_stft
#'
#' @examples
#'
#' 
NULL

# -> sum: ff420e98393249e05692f368d5a1c323 <-
#'
#' @name torch_sum
#'
#' @examples
#'
#' a = torch_randn(c(1, 3))
#' a
#' torch_sum(a)
#'
#'
#' a = torch_randn(c(4, 4))
#' a
#' torch_sum(a, 1)
#' b = torch_arange(4 * 5 * 6)$view(4, 5, 6)
#' torch_sum(b, list(2, 1))
NULL

# -> sqrt: b76885f41f63ece994bbaf0ce467a641 <-
#'
#' @name torch_sqrt
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_sqrt(a)
NULL

# -> std: 7d8af1d6a638e0357a15e0e4c219e01f <-
#'
#' @name torch_std
#'
#' @examples
#'
#' a = torch_randn(c(1, 3))
#' a
#' torch_std(a)
#'
#'
#' a = torch_randn(c(4, 4))
#' a
#' torch_std(a, dim=1)
NULL

# -> std_mean: f4d2c004e60bf36f4215a074cc6be7a8 <-
#'
#' @name torch_std_mean
#'
#' @examples
#'
#' a = torch_randn(c(1, 3))
#' a
#' torch_std_mean(a)
#'
#'
#' a = torch_randn(c(4, 4))
#' a
#' torch_std_mean(a, 1)
NULL

# -> prod: b792cd8dd8a1ddb4f069fd8da55e2cc5 <-
#'
#' @name torch_prod
#'
#' @examples
#'
#' a = torch_randn(c(1, 3))
#' a
#' torch_prod(a)
#'
#'
#' a = torch_randn(c(4, 2))
#' a
#' torch_prod(a, 1)
NULL

# -> t: 466a823d3577b9b648a30c710a666961 <-
#'
#' @name torch_t
#'
#' @examples
#'
#' x = torch_randn(())
#' x
#' torch_t(x)
#' x = torch_randn(c(3))
#' x
#' torch_t(x)
#' x = torch_randn(c(2, 3))
#' x
#' torch_t(x)
NULL

# -> tan: 8714961810e22d139c5a9faa51067d0f <-
#'
#' @name torch_tan
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_tan(a)
NULL

# -> tanh: 1857c578567d80ce651197aa6a24d0e4 <-
#'
#' @name torch_tanh
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_tanh(a)
NULL

# -> tensordot: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_tensordot
#'
#' @examples
#'
#' 
NULL

# -> threshold_: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_threshold_
#'
#' @examples
#'
#' 
NULL

# -> transpose: 2ad5294ea1d6dffa747e0df4a71aaf89 <-
#'
#' @name torch_transpose
#'
#' @examples
#'
#' x = torch_randn(c(2, 3))
#' x
#' torch_transpose(x, 0, 1)
NULL

# -> flip: f904826dc5bd710086a9dc2909395334 <-
#'
#' @name torch_flip
#'
#' @examples
#'
#' x = torch_arange(8)$view(2, 2, 2)
#' x
#' torch_flip(x, c(0, 1))
NULL

# -> roll: 00755e399c0dc98eda1f5d89f4aafa4c <-
#'
#' @name torch_roll
#'
#' @examples
#'
#' x = torch_tensor(c(1, 2, 3, 4, 5, 6, 7, 8))$view(4, 2)
#' x
#' torch_roll(x, 1, 0)
#' torch_roll(x, -1, 0)
#' torch_roll(x, shifts=list(2, 1), dims=list(0, 1))
NULL

# -> rot90: a4904a61edbe4937c7e6f9759fd0188c <-
#'
#' @name torch_rot90
#'
#' @examples
#'
#' x = torch_arange(4)$view(2, 2)
#' x
#' torch_rot90list(x, 1, c(0, 1))
#' x = torch_arange(8)$view(2, 2, 2)
#' x
#' torch_rot90list(x, 1, c(1, 2))
NULL

# -> trapz: 2da24981fd6a01bd6aca237448fca81f <-
#'
#' @name torch_trapz
#'
#' @examples
#'
#' y = torch_randn(list(2, 3))
#' y
#' x = torch_tensor(c([1, 3, 4], [1, 2, 3]))
#' torch_trapz(y, x)
#'
#'
#' 
NULL

# -> trunc: ee5d3a1108e6a41183cdc5329e123e93 <-
#'
#' @name torch_trunc
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_trunc(a)
NULL

# -> unique_consecutive: 3558b04b7d86925537500c6b9e2ec7f7 <-
#'
#' @name torch_unique_consecutive
#'
#' @examples
#'
#' x = torch_tensor(c(1, 1, 2, 2, 3, 1, 1, 2))
#' output = torch_unique_consecutive(x)
#' output
#' output, inverse_indices = torch_unique_consecutive(x, return_inverse=TRUE)
#' output
#' inverse_indices
#' output, counts = torch_unique_consecutive(x, return_counts=TRUE)
#' output
#' counts
NULL

# -> unsqueeze: 5ea03d7559a857db944438afe27144f0 <-
#'
#' @name torch_unsqueeze
#'
#' @examples
#'
#' x = torch_tensor(c(1, 2, 3, 4))
#' torch_unsqueeze(x, 0)
#' torch_unsqueeze(x, 1)
NULL

# -> var: ffcbc64ae608322ab3a81abb9bd6ca79 <-
#'
#' @name torch_var
#'
#' @examples
#'
#' a = torch_randn(c(1, 3))
#' a
#' torch_var(a)
#'
#'
#' a = torch_randn(c(4, 4))
#' a
#' torch_var(a, 1)
NULL

# -> var_mean: 7fc6b1cd7b3b3cb5c73ae138198998fc <-
#'
#' @name torch_var_mean
#'
#' @examples
#'
#' a = torch_randn(c(1, 3))
#' a
#' torch_var_mean(a)
#'
#'
#' a = torch_randn(c(4, 4))
#' a
#' torch_var_mean(a, 1)
NULL

# -> where: 77969dcb74c1aedd058a13c288978649 <-
#'
#' @name torch_where
#'
#' @examples
#'
#' x = torch_randn(c(3, 2))
#' y = torch_ones(c(3, 2))
#' x
#' torch_where(x > 0, x, y)
#'
#'
#' 
NULL

# -> zeros: 0020d7561807e3ad278625fe7142bd81 <-
#'
#' @name torch_zeros
#'
#' @examples
#'
#' torch_zeros(c(2, 3))
#' torch_zeros(c(5))
NULL

# -> zeros_like: d0c14415655d05f80a653596649e990b <-
#'
#' @name torch_zeros_like
#'
#' @examples
#'
#' input = torch_empty(2, 3)
#' torch_zeros_like(input)
NULL

# -> norm: 4ab2ce890e5fa2350be2d8eb5fcf568f <-
#'
#' @name torch_norm
#'
#' @examples
#'
#' import torch
#' a = torch_arange(9, dtype= torch_float) - 4
#' b = a$reshape(list(3, 3))
#' torch_norm(a)
#' torch_norm(b)
#' torch_norm(a, float('inf'))
#' torch_norm(b, float('inf'))
#' c = torch_tensor(c([ 1, 2, 3],[-1, 1, 4]) , dtype= torch_float)
#' torch_norm(c, dim=0)
#' torch_norm(c, dim=1)
#' torch_norm(c, p=1, dim=1)
#' d = torch_arange(8, dtype= torch_float)$reshape(2,2,2)
#' torch_norm(d, dim=list(1,2))
#' torch_norm(dc(0, :, :)), torch_norm(dc(1, :, :))
NULL

# -> pow: b92908baeb402c045e0bdff62c5d30aa <-
#'
#' @name torch_pow
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_pow(a, 2)
#' exp = torch_arange(1., 5.)
#' a = torch_arange(1., 5.)
#' a
#' exp
#' torch_pow(a, exp)
#'
#'
#' exp = torch_arange(1., 5.)
#' base = 2
#' torch_pow(base, exp)
NULL

# -> addmm: b16eb2c0271824e00fa861289ad5b484 <-
#'
#' @name torch_addmm
#'
#' @examples
#'
#' M = torch_randn(c(2, 3))
#' mat1 = torch_randn(c(2, 3))
#' mat2 = torch_randn(c(3, 3))
#' torch_addmm(M, mat1, mat2)
NULL

# -> sparse_coo_tensor: 15fd6329a825ca8e82796bc67a9381bf <-
#'
#' @name torch_sparse_coo_tensor
#'
#' @examples
#'
#' i = torch_tensor(c([0, 1, 1),
#' v = torch_tensor(c(3, 4, 5), dtype=torch_float32())
#' torch_sparse_coo_tensor(i, v, c(2, 4))
#' torch_sparse_coo_tensor(i, v)  # Shape inference
#' torch_sparse_coo_tensor(i, v, c(2, 4),
#' S = torch_sparse_coo_tensor(torch_empty(c(1, 0]), [], [1))
#' S = torch_sparse_coo_tensor(torch_empty(c(1, 0)), torch_empty(c(0, 2]), [1, 2))
NULL

# -> unbind: 48a0b7748852254ec8a30b98f2c5daa6 <-
#'
#' @name torch_unbind
#'
#' @examples
#'
#' torch_unbind(torch_tensor(c([1, 2, 3),
#'                            c(4, 5, 6),
#'                            c(7, 8, 9])))
NULL

# -> quantize_per_tensor: 33142780eb40ccc7ca3af7f1f3443f42 <-
#'
#' @name torch_quantize_per_tensor
#'
#' @examples
#'
#' torch_quantize_per_tensor(torch_tensor(c(-1.0, 0.0, 1.0, 2.0)), 0.1, 10, torch_quint8)
#' torch_quantize_per_tensor(torch_tensor(c(-1.0, 0.0, 1.0, 2.0)), 0.1, 10, torch_quint8)$int_repr()
NULL

# -> quantize_per_channel: 717e167e8d0686bdb13a9951b591805f <-
#'
#' @name torch_quantize_per_channel
#'
#' @examples
#'
#' x = torch_tensor(c([-1.0, 0.0], [1.0, 2.0]))
#' torch_quantize_per_channel(x, torch_tensor(c(0.1, 0.01)), torch_tensor(c(10, 0)), 0, torch_quint8)
#' torch_quantize_per_channel(x, torch_tensor(c(0.1, 0.01)), torch_tensor(c(10, 0)), 0, torch_quint8)$int_repr()
NULL

# -> meshgrid: 7579d1c2d9957802c45601241975f00c <-
#'
#' @name torch_meshgrid
#'
#' @examples
#'
#' x = torch_tensor(c(1, 2, 3))
#' y = torch_tensor(c(4, 5, 6))
#' grid_x, grid_y = torch_meshgrid(x, y)
#' grid_x
#' grid_y
NULL

# -> cartesian_prod: 75e4a5fd1cef174b88942d482de3631f <-
#'
#' @name torch_cartesian_prod
#'
#' @examples
#'
#' a = c(1, 2, 3)
#' b = c(4, 5)
#' list(itertools$product(a, b))
#' tensor_a = torch_tensor(a)
#' tensor_b = torch_tensor(b)
#' torch_cartesian_prod(tensor_a, tensor_b)
NULL

# -> combinations: 40a0b6616e8465b960480878c2edf19c <-
#'
#' @name torch_combinations
#'
#' @examples
#'
#' a = c(1, 2, 3)
#' list(itertools$combinations(a, r=2))
#' list(itertools$combinations(a, r=3))
#' list(itertools$combinations_with_replacement(a, r=2))
#' tensor_a = torch_tensor(a)
#' torch_combinations(tensor_a)
#' torch_combinations(tensor_a, r=3)
#' torch_combinations(tensor_a, with_replacement=TRUE)
NULL

# -> result_type: 457c93f9852cf95c4bd925dd6da73f9a <-
#'
#' @name torch_result_type
#'
#' @examples
#'
#' torch_result_type(torch_tensor(c(1, 2), dtype=torch_int()), 1.0)
#' torch_result_type(torch_tensor(c(1, 2), dtype=torch_uint8()), torch_tensor(1))
NULL

# -> can_cast: b8ee8211ab596021f31e0b59a7de8961 <-
#'
#' @name torch_can_cast
#'
#' @examples
#'
#' torch_can_cast(torch_double, torch_float)
#' torch_can_cast(torch_float, torch_int)
NULL

# -> promote_types: 936479a5f7bc3567ce5d6d9c7853220c <-
#'
#' @name torch_promote_types
#'
#' @examples
#'
#' torch_promote_types(torch_int32, torch_float32))
#' torch_promote_types(torch_uint8, torch_long)
NULL

# -> bitwise_xor: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_bitwise_xor
#'
#' @examples
#'
#' 
NULL

# -> addbmm: bf493966455d48aed3ce09c5d827f1d0 <-
#'
#' @name torch_addbmm
#'
#' @examples
#'
#' M = torch_randn(c(3, 5))
#' batch1 = torch_randn(c(10, 3, 4))
#' batch2 = torch_randn(c(10, 4, 5))
#' torch_addbmm(M, batch1, batch2)
NULL

# -> diag: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_diag
#'
#' @examples
#'
#' 
NULL

# -> cross: 5c132aee22d36da6aab6a348be801aca <-
#'
#' @name torch_cross
#'
#' @examples
#'
#' a = torch_randn(c(4, 3))
#' a
#' b = torch_randn(c(4, 3))
#' b
#' torch_cross(a, b, dim=1)
#' torch_cross(a, b)
NULL

# -> triu: 4332b1ad6ffa185aaff45c27f56d8e0a <-
#'
#' @name torch_triu
#'
#' @examples
#'
#' a = torch_randn(c(3, 3))
#' a
#' torch_triu(a)
#' torch_triu(a, diagonal=1)
#' torch_triu(a, diagonal=-1)
#' b = torch_randn(c(4, 6))
#' b
#' torch_triu(b, diagonal=1)
#' torch_triu(b, diagonal=-1)
NULL

# -> tril: 41eb03dc9264e93b2c74df8d5c8de1b3 <-
#'
#' @name torch_tril
#'
#' @examples
#'
#' a = torch_randn(c(3, 3))
#' a
#' torch_tril(a)
#' b = torch_randn(c(4, 6))
#' b
#' torch_tril(b, diagonal=1)
#' torch_tril(b, diagonal=-1)
NULL

# -> tril_indices: 20cd0fcca0b6191d225bf0abd5094205 <-
#'
#' @name torch_tril_indices
#'
#' @examples
#'
#' a = torch_tril_indices(3, 3)
#' a
#' a = torch_tril_indices(4, 3, -1)
#' a
#' a = torch_tril_indices(4, 3, 1)
#' a
NULL

# -> triu_indices: 76d0b63800a88c072f23f21082747a60 <-
#'
#' @name torch_triu_indices
#'
#' @examples
#'
#' a = torch_triu_indices(3, 3)
#' a
#' a = torch_triu_indices(4, 3, -1)
#' a
#' a = torch_triu_indices(4, 3, 1)
#' a
NULL

# -> trace: 61b80c94b00b39311af82934ffa28527 <-
#'
#' @name torch_trace
#'
#' @examples
#'
#' x = torch_arange(1., 10.)$view(3, 3)
#' x
#' torch_trace(x)
NULL

# -> ne: 986b6728d94699f023780bab7f45b755 <-
#'
#' @name torch_ne
#'
#' @examples
#'
#' torch_ne(torch_tensor(c([1, 2], [3, 4])), torch_tensor(c([1, 1], [4, 4])))
NULL

# -> eq: a33df8003eca79582c8f4a4bb462f71b <-
#'
#' @name torch_eq
#'
#' @examples
#'
#' torch_eq(torch_tensor(c([1, 2], [3, 4])), torch_tensor(c([1, 1], [4, 4])))
NULL

# -> ge: a0bbf4c29678fad9aa9b72e88aa27251 <-
#'
#' @name torch_ge
#'
#' @examples
#'
#' torch_ge(torch_tensor(c([1, 2], [3, 4])), torch_tensor(c([1, 1], [4, 4])))
NULL

# -> le: b86c48a2ade49201678abb6d8ee55cfc <-
#'
#' @name torch_le
#'
#' @examples
#'
#' torch_le(torch_tensor(c([1, 2], [3, 4])), torch_tensor(c([1, 1], [4, 4])))
NULL

# -> gt: 04e8640d3d9267e822c74b2f6106eb61 <-
#'
#' @name torch_gt
#'
#' @examples
#'
#' torch_gt(torch_tensor(c([1, 2], [3, 4])), torch_tensor(c([1, 1], [4, 4])))
NULL

# -> lt: c7eb35966f593463d6be264bb11d7193 <-
#'
#' @name torch_lt
#'
#' @examples
#'
#' torch_lt(torch_tensor(c([1, 2], [3, 4])), torch_tensor(c([1, 1], [4, 4])))
NULL

# -> take: 6e7180b10b856447bdb1ef0207afe0e6 <-
#'
#' @name torch_take
#'
#' @examples
#'
#' src = torch_tensor(c([4, 3, 5),
#' torch_take(src, torch_tensor(c(0, 2, 5)))
NULL

# -> index_select: bb196a5a712a27ff8c564f2c58fd73e9 <-
#'
#' @name torch_index_select
#'
#' @examples
#'
#' x = torch_randn(c(3, 4))
#' x
#' indices = torch_tensor(c(0, 2))
#' torch_index_select(x, 0, indices)
#' torch_index_select(x, 1, indices)
NULL

# -> masked_select: 618a8ef1cbdcacc08bdb8feb9ce85a1e <-
#'
#' @name torch_masked_select
#'
#' @examples
#'
#' x = torch_randn(c(3, 4))
#' x
#' mask = x$ge(0.5)
#' mask
#' torch_masked_select(x, mask)
NULL

# -> nonzero: 64ac786ea85b2c78b0b026f0bacf4743 <-
#'
#' @name torch_nonzero
#'
#' @examples
#'
#' torch_nonzero(torch_tensor(c(1, 1, 1, 0, 1)))
#' torch_nonzero(torch_tensor(c([0.6, 0.0, 0.0, 0.0),
#' torch_nonzero(torch_tensor(c(1, 1, 1, 0, 1)), as_tuple=TRUE)
#' torch_nonzero(torch_tensor(c([0.6, 0.0, 0.0, 0.0),
#' torch_nonzero(torch_tensor(5), as_tuple=TRUE)
NULL

# -> gather: 8b78555961eb64349f87947472a2b135 <-
#'
#' @name torch_gather
#'
#' @examples
#'
#' t = torch_tensor(c([1,2],[3,4]))
#' torch_gather(t, 1, torch_tensor(c([0,0],[1,0])))
NULL

# -> addcmul: 6a8d489cb7838a0068b6e66d827c9b57 <-
#'
#' @name torch_addcmul
#'
#' @examples
#'
#' t = torch_randn(c(1, 3))
#' t1 = torch_randn(c(3, 1))
#' t2 = torch_randn(c(1, 3))
#' torch_addcmul(t, 0.1, t1, t2)
NULL

# -> addcdiv: b8757adc5a9f772c5eb81edbfd542b30 <-
#'
#' @name torch_addcdiv
#'
#' @examples
#'
#' t = torch_randn(c(1, 3))
#' t1 = torch_randn(c(3, 1))
#' t2 = torch_randn(c(1, 3))
#' torch_addcdiv(t, 0.1, t1, t2)
NULL

# -> lstsq: 023890b90fad2168bd32ee5acb32c8ab <-
#'
#' @name torch_lstsq
#'
#' @examples
#'
#' A = torch_tensor(c([1., 1, 1),
#' B = torch_tensor(c([-10., -3),
#' X, _ = torch_lstsq(B, A)
#' X
NULL

# -> triangular_solve: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_triangular_solve
#'
#' @examples
#'
#' 
NULL

# -> symeig: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_symeig
#'
#' @examples
#'
#' 
NULL

# -> eig: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_eig
#'
#' @examples
#'
#' 
NULL

# -> svd: f7e9c76244c545b0d618e9ab56b997e0 <-
#'
#' @name torch_svd
#'
#' @examples
#'
#' a = torch_randn(c(5, 3))
#' a
#' u, s, v = torch_svd(a)
#' u
#' s
#' v
#' torch_dist(a, torch_mm(torch_mm(u, torch_diag(s)), v$t()))
#' a_big = torch_randn(c(7, 5, 3))
#' u, s, v = torch_svd(a_big)
#' torch_dist(a_big, torch_matmul(torch_matmul(u, torch_diag_embed(s)), v$transpose(-2, -1)))
NULL

# -> cholesky: f9842720c4f2766eceec4ecd5f0e5170 <-
#'
#' @name torch_cholesky
#'
#' @examples
#'
#' a = torch_randn(c(3, 3))
#' a = torch_mm(a, a$t()) # make symmetric positive-definite
#' l = torch_cholesky(a)
#' a
#' l
#' torch_mm(l, l$t())
#' a = torch_randn(c(3, 2, 2))
#' a = torch_matmul(a, a$transpose(-1, -2)) + 1e-03 # make symmetric positive-definite
#' l = torch_cholesky(a)
#' z = torch_matmul(l, l$transpose(-1, -2))
#' torch_max(torch_abs(z - a)) # Max non-zero
NULL

# -> cholesky_solve: e085a866ed7458ff15b3740d33e8be63 <-
#'
#' @name torch_cholesky_solve
#'
#' @examples
#'
#' a = torch_randn(c(3, 3))
#' a = torch_mm(a, a$t()) # make symmetric positive definite
#' u = torch_cholesky(a)
#' a
#' b = torch_randn(c(3, 2))
#' b
#' torch_cholesky_solve(b, u)
#' torch_mm(a$inverse(), b)
NULL

# -> solve: 3ffb4bec14cc18d40d261d75fb1236de <-
#'
#' @name torch_solve
#'
#' @examples
#'
#' A = torch_tensor(c([6.80, -2.11,  5.66,  5.97,  8.23),
#' B = torch_tensor(c([4.02,  6.19, -8.22, -7.57, -3.03),
#' X, LU = torch_solve(B, A)
#' torch_dist(B, torch_mm(A, X))
#' # Batched solver example
#' A = torch_randn(c(2, 3, 1, 4, 4))
#' B = torch_randn(c(2, 3, 1, 4, 6))
#' X, LU = torch_solve(B, A)
#' torch_dist(B, A$matmul(X))
NULL

# -> cholesky_inverse: 311df498754dd47071821358bba55bf0 <-
#'
#' @name torch_cholesky_inverse
#'
#' @examples
#'
#' a = torch_randn(c(3, 3))
#' a = torch_mm(a, a$t()) + 1e-05 * torch_eye(3) # make symmetric positive definite
#' u = torch_cholesky(a)
#' a
#' torch_cholesky_inverse(u)
#' a$inverse()
NULL

# -> qr: eb384010e7e5a654c01cb5d09cf5c43c <-
#'
#' @name torch_qr
#'
#' @examples
#'
#' a = torch_tensor(c([12., -51, 4], [6, 167, -68], [-4, 24, -41]))
#' q, r = torch_qr(a)
#' q
#' r
#' torch_mm(q, r)$round()
#' torch_mm(q$t(), q)$round()
#' a = torch_randn(c(3, 4, 5))
#' q, r = torch_qr(a, some=FALSE)
#' torch_allclose(torch_matmul(q, r), a)
#' torch_allclose(torch_matmul(q$transpose(-2, -1), q), torch_eye(5))
NULL

# -> geqrf: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_geqrf
#'
#' @examples
#'
#' 
NULL

# -> orgqr: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_orgqr
#'
#' @examples
#'
#' 
NULL

# -> ormqr: 3558892c5062f8d69312142540f3b3ab <-
#'
#' @name torch_ormqr
#'
#' @examples
#'
#' 
NULL

# -> lu_solve: 006d84a7d66969b7cd66c3a0fdb26748 <-
#'
#' @name torch_lu_solve
#'
#' @examples
#'
#' A = torch_randn(c(2, 3, 3))
#' b = torch_randn(c(2, 3, 1))
#' A_LU = torch_lu(A)
#' x = torch_lu_solve(b, *A_LU)
#' torch_norm(torch_bmm(A, x) - b)
NULL

# -> multinomial: 40effdd493dff97f74f918666d87ff47 <-
#'
#' @name torch_multinomial
#'
#' @examples
#'
#' weights = torch_tensor(c(0, 10, 3, 0), dtype=torch_float()) # create a tensor of weights
#' torch_multinomial(weights, 2)
#' torch_multinomial(weights, 4) # ERROR!
#' torch_multinomial(weights, 4, replacement=TRUE)
NULL

# -> lgamma: 527a28cc1f5c1d323c75d2c0ad1e6ced <-
#'
#' @name torch_lgamma
#'
#' @examples
#'
#' a = torch_arange(0.5, 2, 0.5)
#' torch_lgamma(a)
NULL

# -> digamma: de689631090acc6857392340a027da2a <-
#'
#' @name torch_digamma
#'
#' @examples
#'
#' a = torch_tensor(c(1, 0.5))
#' torch_digamma(a)
NULL

# -> polygamma: 825d64aa95b39404becdbd325b4ef39c <-
#'
#' @name torch_polygamma
#'
#' @examples
#'
#' a = torch_tensor(c(1, 0.5))
#' torch_polygamma(1, a)
NULL

# -> erfinv: 254273796a993a457382afb8498c126f <-
#'
#' @name torch_erfinv
#'
#' @examples
#'
#' torch_erfinv(torch_tensor(c(0, 0.5, -1.)))
NULL

# -> sign: c7c2525aa236fd6e4bbd1a54c06eab1f <-
#'
#' @name torch_sign
#'
#' @examples
#'
#' a = torch_tensor(c(0.7, -1.2, 0., 2.3))
#' a
#' torch_sign(a)
NULL

# -> dist: 75e000e7aae6baea576492ba6b6c1749 <-
#'
#' @name torch_dist
#'
#' @examples
#'
#' x = torch_randn(c(4))
#' x
#' y = torch_randn(c(4))
#' y
#' torch_dist(x, y, 3.5)
#' torch_dist(x, y, 3)
#' torch_dist(x, y, 0)
#' torch_dist(x, y, 1)
NULL

# -> atan2: c32d8631c7895cb730295d27a72f85d0 <-
#'
#' @name torch_atan2
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_atan2list(a, torch_randn(c(4)))
NULL

# -> lerp: 71d61510715131059e4e2699a61bd6f4 <-
#'
#' @name torch_lerp
#'
#' @examples
#'
#' start = torch_arange(1., 5.)
#' end = torch_empty(4)$fill_list(10)
#' start
#' end
#' torch_lerp(start, end, 0.5)
#' torch_lerp(start, end, torch_full_like(start, 0.5))
NULL

# -> histc: f5ce23581f8b8d8501080e28dbe71eb3 <-
#'
#' @name torch_histc
#'
#' @examples
#'
#' torch_histc(torch_tensor(c(1., 2, 1)), bins=4, min=0, max=3)
NULL

# -> fmod: 88f7054e335c8d19c413834d5a403c17 <-
#'
#' @name torch_fmod
#'
#' @examples
#'
#' torch_fmod(torch_tensor(c(-3., -2, -1, 1, 2, 3)), 2)
#' torch_fmod(torch_tensor(c(1., 2, 3, 4, 5)), 1.5)
NULL

# -> remainder: f84b84ba537d8ef796c1e71917f4b171 <-
#'
#' @name torch_remainder
#'
#' @examples
#'
#' torch_remainder(torch_tensor(c(-3., -2, -1, 1, 2, 3)), 2)
#' torch_remainder(torch_tensor(c(1., 2, 3, 4, 5)), 1.5)
NULL

# -> sort: 3cef707b9b50dfa62d147421077aaa24 <-
#'
#' @name torch_sort
#'
#' @examples
#'
#' x = torch_randn(c(3, 4))
#' sorted, indices = torch_sort(x)
#' sorted
#' indices
#' sorted, indices = torch_sort(x, 0)
#' sorted
#' indices
NULL

# -> argsort: 8d75a2a9c4f49faecb0d0db9979a4b5b <-
#'
#' @name torch_argsort
#'
#' @examples
#'
#' a = torch_randn(c(4, 4))
#' a
#' torch_argsort(a, dim=1)
NULL

# -> topk: 738e3ac71d3ef585c98097335da6cc66 <-
#'
#' @name torch_topk
#'
#' @examples
#'
#' x = torch_arange(1., 6.)
#' x
#' torch_topk(x, 3)
NULL

# -> renorm: 948e8829296cae446ef294eef9c12955 <-
#'
#' @name torch_renorm
#'
#' @examples
#'
#' x = torch_ones(c(3, 3))
#' xc(1)$fill_list(2)
#' xc(2)$fill_list(3)
#' x
#' torch_renorm(x, 1, 0, 5)
NULL

# -> equal: 36d27f003eedfee7b69488720f12d2b1 <-
#'
#' @name torch_equal
#'
#' @examples
#'
#' torch_equal(torch_tensor(c(1, 2)), torch_tensor(c(1, 2)))
NULL

# -> normal: 722fa249b7feefdb102097b37bb9840d <-
#'
#' @name torch_normal
#'
#' @examples
#'
#' torch_normal(mean=torch_arange(1., 11.), std=torch_arange(1, 0, -0.1))
#'
#'
#' torch_normal(mean=0.5, std=torch_arange(1., 6.))
#'
#'
#' torch_normal(mean=torch_arange(1., 6.))
#'
#'
#' torch_normal(2, 3, size=list(1, 4))
NULL

# -> isfinite: b34f475941aebcaa05417313984a6f58 <-
#'
#' @name torch_isfinite
#'
#' @examples
#'
#' torch_isfinite(torch_tensor([1, float('inf'), 2, float('-inf'), float('nan')]))
NULL