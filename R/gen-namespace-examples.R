

# -> abs: d67751a436577d634e3c777a853aca91 <-
#'
#' @name torch_abs
#'
#' @examples
#'
#' torch_abs(torch_tensor(c(-1, -2, 3)))
NULL
# -> abs <-

# -> angle: 858a1adaf7a0b5bd09953dbbf70376dd <-
#'
#' @name torch_angle
#'
#' @examples
#' \dontrun{
#' torch_angle(torch_tensor(c(-1 + 1i, -2 + 2i, 3 - 3i)))*180/3.14159
#' }
#' 
NULL
# -> angle <-

# -> real: 2e4771622de23ac64e47f112df9b4e43 <-
#'
#' @name torch_real
#'
#' @examples
#' \dontrun{
#' torch_real(torch_tensor(c(-1 + 1i, -2 + 2i, 3 - 3i)))
#' }
NULL
# -> real <-

# -> imag: d426bdcc68428514ec161db7a8064358 <-
#'
#' @name torch_imag
#'
#' @examples
#' \dontrun{
#' torch_imag(torch_tensor(c(-1 + 1i, -2 + 2i, 3 - 3i)))
#' }
NULL
# -> imag <-

# -> conj: ad66b442c268738f2278da63aa2474ba <-
#'
#' @name torch_conj
#'
#' @examples
#' \dontrun{
#' torch_conj(torch_tensor(c(-1 + 1i, -2 + 2i, 3 - 3i)))
#' }
NULL
# -> conj <-

# -> acos: 2ab027cdf77b7d3ee9add06c085dd0ac <-
#'
#' @name torch_acos
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_acos(a)
NULL
# -> acos <-

# -> avg_pool1d: 943a6632ffecbe645a7a4a43192ee185 <-
#'
#' @name torch_avg_pool1d
#'
#'
#' 
NULL
# -> avg_pool1d <-

# -> adaptive_avg_pool1d: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_adaptive_avg_pool1d
#'
#'
NULL
# -> adaptive_avg_pool1d <-

# -> add: f37c4dbf3ceda27df9137fc81e92e89a <-
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
#' torch_add(a, b)
NULL
# -> add <-

# -> addmv: 83d2c5833f1d727d7f56bc48f7463f5b <-
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
# -> addmv <-

# -> addr: 983b4707a573e2b3efd1fb8713873f9e <-
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
# -> addr <-

# -> allclose: 778248f3567ec9b8ec3002a473c04dbf <-
#'
#' @name torch_allclose
#'
#' @examples
#'
#' torch_allclose(torch_tensor(c(10000., 1e-07)), torch_tensor(c(10000.1, 1e-08)))
#' torch_allclose(torch_tensor(c(10000., 1e-08)), torch_tensor(c(10000.1, 1e-09)))
#' torch_allclose(torch_tensor(c(1.0, NaN)), torch_tensor(c(1.0, NaN)))
#' torch_allclose(torch_tensor(c(1.0, NaN)), torch_tensor(c(1.0, NaN)), equal_nan=TRUE)
NULL
# -> allclose <-

# -> arange: 20cca3a72bc0fbf78059a808f4070c2f <-
#'
#' @name torch_arange
#'
#' @examples
#'
#' torch_arange(start = 0, end = 5)
#' torch_arange(1, 4)
#' torch_arange(1, 2.5, 0.5)
NULL
# -> arange <-

# -> argmax: e3fd429271eec8b076afde32f2efdf87 <-
#'
#' @name torch_argmax
#'
#' @examples
#'
#' \dontrun{
#' a = torch_randn(c(4, 4))
#' a
#' torch_argmax(a)
#' }
#'
#'
#' a = torch_randn(c(4, 4))
#' a
#' torch_argmax(a, dim=1)
NULL
# -> argmax <-

# -> argmin: 3414db8f263b133f52bdde7a3b110445 <-
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
# -> argmin <-

# -> as_strided: 748e2c56a61bed2cbb0a275959890f66 <-
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
#' t
NULL
# -> as_strided <-

# -> asin: 809434925e6b1da5e9a818566ae0c99d <-
#'
#' @name torch_asin
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_asin(a)
NULL
# -> asin <-

# -> atan: a835ea6b64d9edaf523d2e2035874ea0 <-
#'
#' @name torch_atan
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_atan(a)
NULL
# -> atan <-

# -> baddbmm: 71b5d1eee47355a237e06fea2b98178e <-
#'
#' @name torch_baddbmm
#'
#' @examples
#'
#' M = torch_randn(c(10, 3, 5))
#' batch1 = torch_randn(c(10, 3, 4))
#' batch2 = torch_randn(c(10, 4, 5))
#' torch_baddbmm(M, batch1, batch2)
NULL
# -> baddbmm <-

# -> bartlett_window: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_bartlett_window
#'
#'
NULL
# -> bartlett_window <-

# -> bernoulli: 108c25f843728b828911fb500fa4cc1e <-
#'
#' @name torch_bernoulli
#'
#' @examples
#'
#' a = torch_empty(c(3, 3))$uniform_(0, 1)  # generate a uniform random matrix with range c(0, 1)
#' a
#' torch_bernoulli(a)
#' a = torch_ones(c(3, 3)) # probability of drawing "1" is 1
#' torch_bernoulli(a)
#' a = torch_zeros(c(3, 3)) # probability of drawing "1" is 0
#' torch_bernoulli(a)
NULL
# -> bernoulli <-

# -> bincount: 5bda0cdd59c9a9040efbb2be131f7294 <-
#'
#' @name torch_bincount
#'
#' @examples
#'
#' input = torch_randint(0, 8, list(5), dtype=torch_int64())
#' weights = torch_linspace(0, 1, steps=5)
#' input
#' weights
#' torch_bincount(input, weights)
#' input$bincount(weights)
NULL
# -> bincount <-

# -> bitwise_not: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_bitwise_not
#'
#'
NULL
# -> bitwise_not <-

# -> logical_not: e5721fde5523e719dd89518a27395757 <-
#'
#' @name torch_logical_not
#'
#' @examples
#'
#' torch_logical_not(torch_tensor(c(TRUE, FALSE)))
#' torch_logical_not(torch_tensor(c(0, 1, -10), dtype=torch_int8()))
#' torch_logical_not(torch_tensor(c(0., 1.5, -10.), dtype=torch_double()))
NULL
# -> logical_not <-

# -> logical_xor: 947a25a9349705413ad7723681b625c9 <-
#'
#' @name torch_logical_xor
#'
#' @examples
#'
#' torch_logical_xor(torch_tensor(c(TRUE, FALSE, TRUE)), torch_tensor(c(TRUE, FALSE, FALSE)))
#' a = torch_tensor(c(0, 1, 10, 0), dtype=torch_int8())
#' b = torch_tensor(c(4, 0, 1, 0), dtype=torch_int8())
#' torch_logical_xor(a, b)
#' torch_logical_xor(a$to(dtype=torch_double()), b$to(dtype=torch_double()))
#' torch_logical_xor(a$to(dtype=torch_double()), b)
NULL
# -> logical_xor <-

# -> blackman_window: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_blackman_window
#'
#'
NULL
# -> blackman_window <-

# -> bmm: b7a9569596d0289ef982d49fb198cdf2 <-
#'
#' @name torch_bmm
#'
#' @examples
#'
#' input = torch_randn(c(10, 3, 4))
#' mat2 = torch_randn(c(10, 4, 5))
#' res = torch_bmm(input, mat2)
#' res
NULL
# -> bmm <-

# -> broadcast_tensors: dcecc783135dafb8a5c66509094bd1f1 <-
#'
#' @name torch_broadcast_tensors
#'
#' @examples
#'
#' x = torch_arange(0, 3)$view(c(1, 3))
#' y = torch_arange(0, 2)$view(c(2, 1))
#' out = torch_broadcast_tensors(list(x, y))
#' out[[1]]
NULL
# -> broadcast_tensors <-

# -> cat: 062ab1fbc6fc255601fff1b0916ae2f2 <-
#'
#' @name torch_cat
#'
#' @examples
#'
#' x = torch_randn(c(2, 3))
#' x
#' torch_cat(list(x, x, x), 1)
#' torch_cat(list(x, x, x), 2)
NULL
# -> cat <-

# -> ceil: 1301bf94c5affb34ae2ac84bc039aad0 <-
#'
#' @name torch_ceil
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_ceil(a)
NULL
# -> ceil <-

# -> chain_matmul: 0ceea8ee82228e770cf45a9e77f87caf <-
#'
#' @name torch_chain_matmul
#'
#' @examples
#'
#' a = torch_randn(c(3, 4))
#' b = torch_randn(c(4, 5))
#' c = torch_randn(c(5, 6))
#' d = torch_randn(c(6, 7))
#' torch_chain_matmul(list(a, b, c, d))
NULL
# -> chain_matmul <-

# -> chunk: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_chunk
#'
#'
NULL
# -> chunk <-

# -> clamp: b8715e8ea7b218126f0d0324d38f435b <-
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
# -> clamp <-

# -> conv1d: 3a9090d974ee61599fba1c1c9f3e1d14 <-
#'
#' @name torch_conv1d
#'
#' @examples
#'
#' filters = torch_randn(c(33, 16, 3))
#' inputs = torch_randn(c(20, 16, 50))
#' nnf_conv1d(inputs, filters)
NULL
# -> conv1d <-

# -> conv2d: 73bd1e08715f8f25547a70fb821a6eea <-
#'
#' @name torch_conv2d
#'
#' @examples
#'
#' # With square kernels and equal stride
#' filters = torch_randn(c(8,4,3,3))
#' inputs = torch_randn(c(1,4,5,5))
#' nnf_conv2d(inputs, filters, padding=1)
NULL
# -> conv2d <-

# -> conv3d: f644490bb91b9559df9913760cd99203 <-
#'
#' @name torch_conv3d
#'
#' @examples
#'
#' # filters = torch_randn(c(33, 16, 3, 3, 3))
#' # inputs = torch_randn(c(20, 16, 50, 10, 20))
#' # nnf_conv3d(inputs, filters)
NULL
# -> conv3d <-

# -> conv_tbc: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_conv_tbc
#'
#'
NULL
# -> conv_tbc <-

# -> conv_transpose1d: 69815461ca09463f23c426d0529d9611 <-
#'
#' @name torch_conv_transpose1d
#'
#' @examples
#'
#' inputs = torch_randn(c(20, 16, 50))
#' weights = torch_randn(c(16, 33, 5))
#' nnf_conv_transpose1d(inputs, weights)
NULL
# -> conv_transpose1d <-

# -> conv_transpose2d: 599038ae972c64325923c50172c1c083 <-
#'
#' @name torch_conv_transpose2d
#'
#' @examples
#'
#' # With square kernels and equal stride
#' inputs = torch_randn(c(1, 4, 5, 5))
#' weights = torch_randn(c(4, 8, 3, 3))
#' nnf_conv_transpose2d(inputs, weights, padding=1)
NULL
# -> conv_transpose2d <-

# -> conv_transpose3d: 4699ff7e852ff50d9a34e456612f461d <-
#'
#' @name torch_conv_transpose3d
#'
#' @examples
#' \dontrun{
#' inputs = torch_randn(c(20, 16, 50, 10, 20))
#' weights = torch_randn(c(16, 33, 3, 3, 3))
#' nnf_conv_transpose3d(inputs, weights)
#' }
NULL
# -> conv_transpose3d <-

# -> cos: 187369ee9e12f250c400a79ab1380184 <-
#'
#' @name torch_cos
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_cos(a)
NULL
# -> cos <-

# -> cosh: b80ef04beab2b73f83cea6e79cba081f <-
#'
#' @name torch_cosh
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_cosh(a)
NULL
# -> cosh <-

# -> cumsum: f6094a9c0e6aa88b4e5ae3e7c484c96a <-
#'
#' @name torch_cumsum
#'
#' @examples
#'
#' a = torch_randn(c(10))
#' a
#' torch_cumsum(a, dim=1)
NULL
# -> cumsum <-

# -> cumprod: 9e8797a28630ef38431909cf4523ccb7 <-
#'
#' @name torch_cumprod
#'
#' @examples
#'
#' a = torch_randn(c(10))
#' a
#' torch_cumprod(a, dim=1)
NULL
# -> cumprod <-

# -> det: 9ceaddcc0529d47bc4610b7a0aa04346 <-
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
# -> det <-

# -> diag_embed: d1fa41a75f777a11382c295654caebb8 <-
#'
#' @name torch_diag_embed
#'
#' @examples
#'
#' a = torch_randn(c(2, 3))
#' torch_diag_embed(a)
#' torch_diag_embed(a, offset=1, dim1=1, dim2=3)
NULL
# -> diag_embed <-

# -> diagflat: 643fc76e4753dba5e752cb32f1fbc1ef <-
#'
#' @name torch_diagflat
#'
#' @examples
#'
#' a = torch_randn(c(3))
#' a
#' torch_diagflat(a)
#' torch_diagflat(a, 1)
#' a = torch_randn(c(2, 2))
#' a
#' torch_diagflat(a)
NULL
# -> diagflat <-

# -> diagonal: c9dbc6ad1a03702b45963f0e23f93db3 <-
#'
#' @name torch_diagonal
#'
#' @examples
#'
#' a = torch_randn(c(3, 3))
#' a
#' torch_diagonal(a, offset = 0)
#' torch_diagonal(a, offset = 1)
#' x = torch_randn(c(2, 5, 4, 2))
#' torch_diagonal(x, offset=-1, dim1=1, dim2=2)
NULL
# -> diagonal <-

# -> div: 0d44b6c1c7d9ef55370b4d6ac8800a21 <-
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
# -> div <-

# -> dot: 403246ca3b8964ee4417df21198f74ae <-
#'
#' @name torch_dot
#'
#' @examples
#'
#' torch_dot(torch_tensor(c(2, 3)), torch_tensor(c(2, 1)))
NULL
# -> dot <-

# -> einsum: 26243b32dac1513907e66bad26c21399 <-
#'
#' @name torch_einsum
#'
#' @examples
#' 
#' if (FALSE) {
#'
#' x = torch_randn(c(5))
#' y = torch_randn(c(4))
#' torch_einsum('i,j->ij', list(x, y))  # outer product
#' A = torch_randn(c(3,5,4))
#' l = torch_randn(c(2,5))
#' r = torch_randn(c(2,4))
#' torch_einsum('bn,anm,bm->ba', list(l, A, r)) # compare torch_nn$functional$bilinear
#' As = torch_randn(c(3,2,5))
#' Bs = torch_randn(c(3,5,4))
#' torch_einsum('bij,bjk->bik', list(As, Bs)) # batch matrix multiplication
#' A = torch_randn(c(3, 3))
#' torch_einsum('ii->i', list(A)) # diagonal
#' A = torch_randn(c(4, 3, 3))
#' torch_einsum('...ii->...i', list(A)) # batch diagonal
#' A = torch_randn(c(2, 3, 4, 5))
#' torch_einsum('...ij->...ji', list(A))$shape # batch permute
#' 
#' }
NULL
# -> einsum <-

# -> empty: fc1319cd474c41b5f0544e2df95e87a2 <-
#'
#' @name torch_empty
#'
#' @examples
#'
#' torch_empty(c(2, 3))
NULL
# -> empty <-

# -> empty_like: 57c270b5f820634f47dc2ce3ba5b9a61 <-
#'
#' @name torch_empty_like
#'
#' @examples
#'
#' torch_empty(list(2,3), dtype = torch_int64())
NULL
# -> empty_like <-

# -> empty_strided: a5d33d461b4133d8cb364e4e9a9f5e57 <-
#'
#' @name torch_empty_strided
#'
#' @examples
#'
#' a = torch_empty_strided(list(2, 3), list(1, 2))
#' a
#' a$stride(1)
#' a$size(1)
NULL
# -> empty_strided <-

# -> erf: 4976cdbfc8a05adb26c48eb782b728a4 <-
#'
#' @name torch_erf
#'
#' @examples
#'
#' torch_erf(torch_tensor(c(0, -1., 10.)))
NULL
# -> erf <-

# -> erfc: 6f41b5feb9e66231441913a9bde46a36 <-
#'
#' @name torch_erfc
#'
#' @examples
#'
#' torch_erfc(torch_tensor(c(0, -1., 10.)))
NULL
# -> erfc <-

# -> exp: de21080c698e51ce84e372862e05accc <-
#'
#' @name torch_exp
#'
#' @examples
#'
#' torch_exp(torch_tensor(c(0, log(2))))
NULL
# -> exp <-

# -> expm1: 6eb1d24bf334a77043756ba548bda1b2 <-
#'
#' @name torch_expm1
#'
#' @examples
#'
#' torch_expm1(torch_tensor(c(0, log(2))))
NULL
# -> expm1 <-

# -> eye: 7b16a282d66b8d7af0f412eb09346da0 <-
#'
#' @name torch_eye
#'
#' @examples
#'
#' torch_eye(3)
NULL
# -> eye <-

# -> flatten: ab7be9f73864c6c5274a1a6364c95bcf <-
#'
#' @name torch_flatten
#'
#' @examples
#'
#' t = torch_tensor(matrix(c(1, 2), ncol = 2))
#' torch_flatten(t)
#' torch_flatten(t, start_dim=2)
NULL
# -> flatten <-

# -> floor: b31c152d9e90063827fb0131b02cf41b <-
#'
#' @name torch_floor
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_floor(a)
NULL
# -> floor <-

# -> frac: 753ca2da9acd869db74186fc0c7d0bba <-
#'
#' @name torch_frac
#'
#' @examples
#'
#' torch_frac(torch_tensor(c(1, 2.5, -3.2)))
NULL
# -> frac <-

# -> full: 385ba2d57942fb74ecaff11aad6a36cd <-
#'
#' @name torch_full
#'
#' @examples
#'
#' torch_full(list(2, 3), 3.141592)
NULL
# -> full <-

# -> full_like: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_full_like
#'
#'
NULL
# -> full_like <-

# -> hann_window: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_hann_window
#'
#'
NULL
# -> hann_window <-

# -> hamming_window: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_hamming_window
#'
#'
NULL
# -> hamming_window <-

# -> ger: 7f4239917757355b13e06da6ebcf56b5 <-
#'
#' @name torch_ger
#'
#' @examples
#'
#' v1 = torch_arange(1., 5.)
#' v2 = torch_arange(1., 4.)
#' torch_ger(v1, v2)
NULL
# -> ger <-

# -> fft: e1535611cd22ff7cc83737d4b4dc757e <-
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
#' torch_fft(x, 2)
#' 
NULL
# -> fft <-

# -> ifft: 08194883af0b7f6defe5ddf45417c12b <-
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
# -> ifft <-

# -> rfft: 66062b6f0f65544a82d08c253dab9f84 <-
#'
#' @name torch_rfft
#'
#' @examples
#'
#' x = torch_randn(c(5, 5))
#' torch_rfft(x, 2)
#' torch_rfft(x, 2, onesided=FALSE)
NULL
# -> rfft <-

# -> irfft: ddaf76706c8a1ebc40099f04b8fe48be <-
#'
#' @name torch_irfft
#'
#' @examples
#'
#' x = torch_randn(c(4, 4))
#' torch_rfft(x, 2, onesided=TRUE)
#' x = torch_randn(c(4, 5))
#' torch_rfft(x, 2, onesided=TRUE)
#' y = torch_rfft(x, 2, onesided=TRUE)
#' torch_irfft(y, 2, onesided=TRUE, signal_sizes=c(4,5))  # recover x
NULL
# -> irfft <-

# -> inverse: fe05348b6ac0ba0b8c2ea626c9cb6a60 <-
#'
#' @name torch_inverse
#'
#' @examples
#' \dontrun{
#' x = torch_rand(c(4, 4))
#' y = torch_inverse(x)
#' z = torch_mm(x, y)
#' z
#' torch_max(torch_abs(z - torch_eye(4))) # Max non-zero
#' # Batched inverse example
#' x = torch_randn(c(2, 3, 4, 4))
#' y = torch_inverse(x)
#' z = torch_matmul(x, y)
#' torch_max(torch_abs(z - torch_eye(4)$expand_as(x))) # Max non-zero
#' }
NULL
# -> inverse <-

# -> isnan: 9328fea60b8f3233e4a0cb9c57b77d4f <-
#'
#' @name torch_isnan
#'
#' @examples
#'
#' torch_isnan(torch_tensor(c(1, NaN, 2)))
NULL
# -> isnan <-

# -> is_floating_point: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_is_floating_point
#'
#'
NULL
# -> is_floating_point <-

# -> kthvalue: 1b3dc147ae41919ecb3a6c33be12ca8b <-
#'
#' @name torch_kthvalue
#'
#' @examples
#'
#' x = torch_arange(1., 6.)
#' x
#' torch_kthvalue(x, 4)
#' x=torch_arange(1.,7.)$resize_(c(2,3))
#' x
#' torch_kthvalue(x, 2, 1, TRUE)
NULL
# -> kthvalue <-

# -> linspace: 123864be88056e7381e09bc27976f492 <-
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
# -> linspace <-

# -> log: f7606d81dc184ad41e0f57be6da56ed8 <-
#'
#' @name torch_log
#'
#' @examples
#'
#' a = torch_randn(c(5))
#' a
#' torch_log(a)
NULL
# -> log <-

# -> log10: 57c140ca705ac3090ff11a03ce195897 <-
#'
#' @name torch_log10
#'
#' @examples
#'
#' a = torch_rand(5)
#' a
#' torch_log10(a)
NULL
# -> log10 <-

# -> log1p: efe674bae76df01732ce0293e8c8ebf3 <-
#'
#' @name torch_log1p
#'
#' @examples
#'
#' a = torch_randn(c(5))
#' a
#' torch_log1p(a)
NULL
# -> log1p <-

# -> log2: 3ca065eb4fea9e278f58109cad5ac6c6 <-
#'
#' @name torch_log2
#'
#' @examples
#'
#' a = torch_rand(5)
#' a
#' torch_log2(a)
NULL
# -> log2 <-

# -> logdet: f59a9e35be3196215068611ee0c524e3 <-
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
# -> logdet <-

# -> logspace: 09c08230adfbd9fcab12ae5e4c2e1a97 <-
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
# -> logspace <-

# -> logsumexp: c82688e1cc56c89e807d3a61a67d8b71 <-
#'
#' @name torch_logsumexp
#'
#' @examples
#'
#' a = torch_randn(c(3, 3))
#' torch_logsumexp(a, 1)
NULL
# -> logsumexp <-

# -> matmul: 110288a000325eeed22e5a4c6812a631 <-
#'
#' @name torch_matmul
#'
#' @examples
#'
#' # vector x vector
#' tensor1 = torch_randn(c(3))
#' tensor2 = torch_randn(c(3))
#' torch_matmul(tensor1, tensor2)
#' # matrix x vector
#' tensor1 = torch_randn(c(3, 4))
#' tensor2 = torch_randn(c(4))
#' torch_matmul(tensor1, tensor2)
#' # batched matrix x broadcasted vector
#' tensor1 = torch_randn(c(10, 3, 4))
#' tensor2 = torch_randn(c(4))
#' torch_matmul(tensor1, tensor2)
#' # batched matrix x batched matrix
#' tensor1 = torch_randn(c(10, 3, 4))
#' tensor2 = torch_randn(c(10, 4, 5))
#' torch_matmul(tensor1, tensor2)
#' # batched matrix x broadcasted matrix
#' tensor1 = torch_randn(c(10, 3, 4))
#' tensor2 = torch_randn(c(4, 5))
#' torch_matmul(tensor1, tensor2)
NULL
# -> matmul <-

# -> matrix_rank: 041df6e3dfcfe7a231267bb49b944be8 <-
#'
#' @name torch_matrix_rank
#'
#' @examples
#' 
#' a = torch_eye(10)
#' torch_matrix_rank(a)
NULL
# -> matrix_rank <-

# -> matrix_power: 7c31064980c3fed7ea226c1bcc72b8de <-
#'
#' @name torch_matrix_power
#'
#' @examples
#'
#' a = torch_randn(c(2, 2, 2))
#' a
#' torch_matrix_power(a, 3)
NULL
# -> matrix_power <-

# -> max: c1a333203dfab17cd432e2d9e0d31a30 <-
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
#' torch_max(a, dim = 1)
#'
#'
#' a = torch_randn(c(4))
#' a
#' b = torch_randn(c(4))
#' b
#' torch_max(a, other = b)
NULL
# -> max <-

# -> mean: 0092bec93cb1be5639f5c26d37176426 <-
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
# -> mean <-

# -> median: 59973f7512422de0cab6c363a58768c1 <-
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
# -> median <-

# -> min: 35719b459da328e3214f8ec0cdc4777e <-
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
#' torch_min(a, dim = 1)
#'
#'
#' a = torch_randn(c(4))
#' a
#' b = torch_randn(c(4))
#' b
#' torch_min(a, other = b)
NULL
# -> min <-

# -> mm: 371187e250a019670fbae0dbed04405d <-
#'
#' @name torch_mm
#'
#' @examples
#'
#' mat1 = torch_randn(c(2, 3))
#' mat2 = torch_randn(c(3, 3))
#' torch_mm(mat1, mat2)
NULL
# -> mm <-

# -> mode: 4ac2f0c2be982daf6861ad35f98d69f2 <-
#'
#' @name torch_mode
#'
#' @examples
#'
#' a = torch_randint(0, 50, size = list(5))
#' a
#' torch_mode(a, 1)
NULL
# -> mode <-

# -> mul: 66517b9d221f17098ea48081b599bac7 <-
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
# -> mul <-

# -> mv: 9dfff3e35d31c67763337639379fc1f5 <-
#'
#' @name torch_mv
#'
#' @examples
#'
#' mat = torch_randn(c(2, 3))
#' vec = torch_randn(c(3))
#' torch_mv(mat, vec)
NULL
# -> mv <-

# -> mvlgamma: 17e1d8dc3a9d219efb75758bfbe9a9e2 <-
#'
#' @name torch_mvlgamma
#'
#' @examples
#'
#' a = torch_empty(c(2, 3))$uniform_(1, 2)
#' a
#' torch_mvlgamma(a, 2)
NULL
# -> mvlgamma <-

# -> narrow: 8df7395de5a1fe0ed552cd4416a6d9c5 <-
#'
#' @name torch_narrow
#'
#' @examples
#'
#' x = torch_tensor(matrix(c(1:9), ncol = 3, byrow= TRUE))
#' torch_narrow(x, 1, torch_tensor(0L)$sum(dim = 1), 2)
#' torch_narrow(x, 2, torch_tensor(1L)$sum(dim = 1), 2)
NULL
# -> narrow <-

# -> ones: 1d5d18d0f8b2f7a288d69542f58f8167 <-
#'
#' @name torch_ones
#'
#' @examples
#'
#' torch_ones(c(2, 3))
#' torch_ones(c(5))
NULL
# -> ones <-

# -> ones_like: 70e3bc6c9ae2bd040fb322dbe2f09428 <-
#'
#' @name torch_ones_like
#'
#' @examples
#'
#' input = torch_empty(c(2, 3))
#' torch_ones_like(input)
NULL
# -> ones_like <-

# -> cdist: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_cdist
#'
#'
NULL
# -> cdist <-

# -> pdist: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_pdist
#'
#'
NULL
# -> pdist <-

# -> cosine_similarity: f6f66e6ba78bca3dfdeca3b8fd941f36 <-
#'
#' @name torch_cosine_similarity
#'
#' @examples
#'
#' input1 = torch_randn(c(100, 128))
#' input2 = torch_randn(c(100, 128))
#' output = torch_cosine_similarity(input1, input2)
#' output
NULL
# -> cosine_similarity <-

# -> pixel_shuffle: a768693934e6bd930deee49f0153701d <-
#'
#' @name torch_pixel_shuffle
#'
#' @examples
#'
#' input = torch_randn(c(1, 9, 4, 4))
#' output = nnf_pixel_shuffle(input, 3)
#' print(output$size())
NULL
# -> pixel_shuffle <-

# -> pinverse: 498255c51fcda39d937b53caf635fe30 <-
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
# -> pinverse <-

# -> rand: f95ab78fefb383cafd2b4192885fd1be <-
#'
#' @name torch_rand
#'
#' @examples
#'
#' torch_rand(4)
#' torch_rand(c(2, 3))
NULL
# -> rand <-

# -> rand_like: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_rand_like
#'
#'
NULL
# -> rand_like <-

# -> randint: 94d6f44b2cdcc1b02dde215c47c29dfa <-
#'
#' @name torch_randint
#'
#' @examples
#'
#' torch_randint(3, 5, list(3))
#' torch_randint(0, 10, size = list(2, 2))
#' torch_randint(3, 10, list(2, 2))
NULL
# -> randint <-

# -> randint_like: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_randint_like
#'
#'
NULL
# -> randint_like <-

# -> randn: 35c32ff43f04e1a5f8f3cf6445b6d8d0 <-
#'
#' @name torch_randn
#'
#' @examples
#'
#' torch_randn(c(4))
#' torch_randn(c(2, 3))
NULL
# -> randn <-

# -> randn_like: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_randn_like
#'
#'
NULL
# -> randn_like <-

# -> randperm: fb99117e3838e358c0d679eabf182d6e <-
#'
#' @name torch_randperm
#'
#' @examples
#'
#' torch_randperm(4)
NULL
# -> randperm <-

# -> range: 45a02da537b45d436d0fedf0e29a987d <-
#'
#' @name torch_range
#'
#' @examples
#'
#' torch_range(1, 4)
#' torch_range(1, 4, 0.5)
NULL
# -> range <-

# -> reciprocal: acd4500bfdd975c0a8c344292cf8f955 <-
#'
#' @name torch_reciprocal
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_reciprocal(a)
NULL
# -> reciprocal <-

# -> neg: 82adab499b82a5cc571270b31ca2fb81 <-
#'
#' @name torch_neg
#'
#' @examples
#'
#' a = torch_randn(c(5))
#' a
#' torch_neg(a)
NULL
# -> neg <-

# -> repeat_interleave: 13cf46e105605936f55807aa8c5e5cc2 <-
#'
#' @name torch_repeat_interleave
#'
#' @examples
#' \dontrun{
#' x = torch_tensor(c(1, 2, 3))
#' x$repeat_interleave(2)
#' y = torch_tensor(matrix(c(1, 2, 3, 4), ncol = 2, byrow=TRUE))
#' torch_repeat_interleave(y, 2)
#' torch_repeat_interleave(y, 3, dim=1)
#' torch_repeat_interleave(y, torch_tensor(c(1, 2)), dim=1)
#' }
NULL
# -> repeat_interleave <-

# -> reshape: 87c985ba483f47dbf33c5def81926f9a <-
#'
#' @name torch_reshape
#'
#' @examples
#'
#' a = torch_arange(0, 4)
#' torch_reshape(a, list(2, 2))
#' b = torch_tensor(matrix(c(0, 1, 2, 3), ncol = 2, byrow=TRUE))
#' torch_reshape(b, list(-1))
NULL
# -> reshape <-

# -> round: 76fa7a264e0e6fc92826ee4aff8dba57 <-
#'
#' @name torch_round
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_round(a)
NULL
# -> round <-

# -> rrelu_: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_rrelu_
#'
#'
NULL
# -> rrelu_ <-

# -> relu_: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_relu_
#'
#'
NULL
# -> relu_ <-

# -> rsqrt: a5b740997bfdcf6c7aaabd6f1e090ff4 <-
#'
#' @name torch_rsqrt
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_rsqrt(a)
NULL
# -> rsqrt <-

# -> selu_: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_selu_
#'
#'
NULL
# -> selu_ <-

# -> celu_: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_celu_
#'
#'
NULL
# -> celu_ <-

# -> sigmoid: 048db144405d843bf7dce130440f0ceb <-
#'
#' @name torch_sigmoid
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_sigmoid(a)
NULL
# -> sigmoid <-

# -> sin: a3445f69d5ccde3c4642d04f04cd4547 <-
#'
#' @name torch_sin
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_sin(a)
NULL
# -> sin <-

# -> sinh: 0c7b5dfd08bd6473101c8fd667dce2d5 <-
#'
#' @name torch_sinh
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_sinh(a)
NULL
# -> sinh <-

# -> slogdet: d4c8ce541efb53e78ab78ad3e4877866 <-
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
# -> slogdet <-

# -> split: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_split
#'
#'
NULL
# -> split <-

# -> squeeze: 38da86d00c7f192edad4de4c29da2945 <-
#'
#' @name torch_squeeze
#'
#' @examples
#'
#' x = torch_zeros(c(2, 1, 2, 1, 2))
#' x
#' y = torch_squeeze(x)
#' y
#' y = torch_squeeze(x, 1)
#' y
#' y = torch_squeeze(x, 2)
#' y
NULL
# -> squeeze <-

# -> stack: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_stack
#'
#'
NULL
# -> stack <-

# -> stft: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_stft
#'
#'
NULL
# -> stft <-

# -> sum: 4a5dc728b3c8d01fcd333d25edefd056 <-
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
#' b = torch_arange(0, 4 * 5 * 6)$view(c(4, 5, 6))
#' torch_sum(b, list(2, 1))
NULL
# -> sum <-

# -> sqrt: 4403b89ab0f819281ea0a92a194f8735 <-
#'
#' @name torch_sqrt
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_sqrt(a)
NULL
# -> sqrt <-

# -> std: 859760d7444b91c1875b131e43210a54 <-
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
# -> std <-

# -> std_mean: c2fed49efa35511c534adad076427c6b <-
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
# -> std_mean <-

# -> prod: 794e486d1caa56c0be86a2e9dc5efcb6 <-
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
# -> prod <-

# -> t: 4250fd7ed25c43fb92ad9e82d508aed9 <-
#'
#' @name torch_t
#'
#' @examples
#'
#' x = torch_randn(c(2,3))
#' x
#' torch_t(x)
#' x = torch_randn(c(3))
#' x
#' torch_t(x)
#' x = torch_randn(c(2, 3))
#' x
#' torch_t(x)
NULL
# -> t <-

# -> tan: ee053376407ce89e5a13d44ead2370bf <-
#'
#' @name torch_tan
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_tan(a)
NULL
# -> tan <-

# -> tanh: 129dbdcce5c48a41368da2679efda266 <-
#'
#' @name torch_tanh
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_tanh(a)
NULL
# -> tanh <-

# -> tensordot: bba1483fe9598180ce5434984285115b <-
#'
#' @name torch_tensordot
#'
#' @examples
#'
#' a = torch_arange(start = 0, end = 60.)$reshape(c(3, 4, 5))
#' b = torch_arange(start = 0, end = 24.)$reshape(c(4, 3, 2))
#' torch_tensordot(a, b, dims_self=c(2, 1), dims_other = c(1, 2))
#' \dontrun{
#' a = torch_randn(3, 4, 5, device='cuda')
#' b = torch_randn(4, 5, 6, device='cuda')
#' c = torch_tensordot(a, b, dims=2)$cpu()
#' }
NULL
# -> tensordot <-

# -> threshold_: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_threshold_
#'
#'
NULL
# -> threshold_ <-

# -> transpose: 66189c6752878fe1bab38cd45b9f0b44 <-
#'
#' @name torch_transpose
#'
#' @examples
#'
#' x = torch_randn(c(2, 3))
#' x
#' torch_transpose(x, 1, 2)
NULL
# -> transpose <-

# -> flip: d68baa96bb21694841c117a8ae510cc1 <-
#'
#' @name torch_flip
#'
#' @examples
#'
#' x = torch_arange(0, 8)$view(c(2, 2, 2))
#' x
#' torch_flip(x, c(1, 2))
NULL
# -> flip <-

# -> roll: 8b35282a72c8230c8e14b5f51a12194a <-
#'
#' @name torch_roll
#'
#' @examples
#'
#' x = torch_tensor(c(1, 2, 3, 4, 5, 6, 7, 8))$view(c(4, 2))
#' x
#' torch_roll(x, 1, 1)
#' torch_roll(x, -1, 1)
#' torch_roll(x, shifts=list(2, 1), dims=list(1, 2))
NULL
# -> roll <-

# -> rot90: fb7f5152854707942aa716eaa06afe8a <-
#'
#' @name torch_rot90
#'
#' @examples
#'
#' x = torch_arange(0, 4)$view(c(2, 2))
#' x
#' torch_rot90(x, 1, c(1, 2))
#' x = torch_arange(0, 8)$view(c(2, 2, 2))
#' x
#' torch_rot90(x, 1, c(1, 2))
NULL
# -> rot90 <-

# -> trapz: 17b62c0d13522f4dcb33b03e74e29bc1 <-
#'
#' @name torch_trapz
#'
#' @examples
#'
#' y = torch_randn(list(2, 3))
#' y
#' x = torch_tensor(matrix(c(1, 3, 4, 1, 2, 3), ncol = 3, byrow=TRUE))
#' torch_trapz(y, x = x)
#'
NULL
# -> trapz <-

# -> trunc: f245b418351aa4c3cc8798f09aafee6a <-
#'
#' @name torch_trunc
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_trunc(a)
NULL
# -> trunc <-

# -> unique_consecutive: 9f5ea720f3a13719d75ef4bc2fc4004f <-
#'
#' @name torch_unique_consecutive
#'
#' @examples
#' x = torch_tensor(c(1, 1, 2, 2, 3, 1, 1, 2))
#' output = torch_unique_consecutive(x)
#' output
#' torch_unique_consecutive(x, return_inverse=TRUE)
#' torch_unique_consecutive(x, return_counts=TRUE)
NULL
# -> unique_consecutive <-

# -> unsqueeze: 5c1b8ad792a08bb7ab6cd157d078f6b1 <-
#'
#' @name torch_unsqueeze
#'
#' @examples
#'
#' x = torch_tensor(c(1, 2, 3, 4))
#' torch_unsqueeze(x, 1)
#' torch_unsqueeze(x, 2)
NULL
# -> unsqueeze <-

# -> var: b0817af308e4b10f4ef1bd402cb128ea <-
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
# -> var <-

# -> var_mean: 2a4e6cdbfe967216d7efdd628d179889 <-
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
# -> var_mean <-

# -> where: 8916a6fb35031f8f9eeeeba4d59dc464 <-
#'
#' @name torch_where
#'
#' @examples
#'
#' \dontrun{
#' x = torch_randn(c(3, 2))
#' y = torch_ones(c(3, 2))
#' x
#' torch_where(x > 0, x, y)
#' }
#'
#'
#' 
NULL
# -> where <-

# -> zeros: 7721a7c7aac9c797e272f299fd2a57fb <-
#'
#' @name torch_zeros
#'
#' @examples
#'
#' torch_zeros(c(2, 3))
#' torch_zeros(c(5))
NULL
# -> zeros <-

# -> zeros_like: 075019bd5dd6e47910361930abff6593 <-
#'
#' @name torch_zeros_like
#'
#' @examples
#'
#' input = torch_empty(c(2, 3))
#' torch_zeros_like(input)
NULL
# -> zeros_like <-

# -> norm: 631c4b122bcf19cc961d74b56d4a527b <-
#'
#' @name torch_norm
#'
#' @examples
#' 
#' a = torch_arange(0, 9, dtype = torch_float())
#' b = a$reshape(list(3, 3))
#' torch_norm(a)
#' torch_norm(b)
#' torch_norm(a, Inf)
#' torch_norm(b, Inf)
#' 
NULL
# -> norm <-

# -> pow: d4402c27f8c224e56d20a232c1eac291 <-
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
# -> pow <-

# -> addmm: d9ab8b42264729f3654e7d6b30b11b09 <-
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
# -> addmm <-

# -> sparse_coo_tensor: 89c4a3054942381686d7a96fb8161134 <-
#'
#' @name torch_sparse_coo_tensor
#'
#' @examples
#'
#' i = torch_tensor(matrix(c(1, 2, 2, 3, 1, 3), ncol = 3, byrow = TRUE), dtype=torch_int64())
#' v = torch_tensor(c(3, 4, 5), dtype=torch_float32())
#' torch_sparse_coo_tensor(i, v)
#' torch_sparse_coo_tensor(i, v, c(2, 4))
#' 
#' # create empty sparse tensors
#' S = torch_sparse_coo_tensor(
#'   torch_empty(c(1, 0), dtype = torch_int64()), 
#'   torch_tensor(numeric(), dtype = torch_float32()), 
#'   c(1)
#' )
#' S = torch_sparse_coo_tensor(
#'   torch_empty(c(1, 0), dtype = torch_int64()), 
#'   torch_empty(c(0, 2)), 
#'   c(1, 2)
#' )
NULL
# -> sparse_coo_tensor <-

# -> unbind: 5950592b8a9a3320a4ca6432754f5ebd <-
#'
#' @name torch_unbind
#'
#' @examples
#'
#' torch_unbind(torch_tensor(matrix(1:9, ncol = 3, byrow=TRUE)))
NULL
# -> unbind <-

# -> quantize_per_tensor: 32638b3b40231e2fd3a25e68abd27834 <-
#'
#' @name torch_quantize_per_tensor
#'
#' @examples
#' torch_quantize_per_tensor(torch_tensor(c(-1.0, 0.0, 1.0, 2.0)), 0.1, 10, torch_quint8())
#' torch_quantize_per_tensor(torch_tensor(c(-1.0, 0.0, 1.0, 2.0)), 0.1, 10, torch_quint8())$int_repr()
NULL
# -> quantize_per_tensor <-

# -> quantize_per_channel: 9ad2eb8b48ffc73bf83c390027ad3313 <-
#'
#' @name torch_quantize_per_channel
#'
#' @examples
#' x = torch_tensor(matrix(c(-1.0, 0.0, 1.0, 2.0), ncol = 2, byrow = TRUE))
#' torch_quantize_per_channel(x, torch_tensor(c(0.1, 0.01)), 
#'                            torch_tensor(c(10L, 0L)), 0, torch_quint8())
#' torch_quantize_per_channel(x, torch_tensor(c(0.1, 0.01)), 
#'                            torch_tensor(c(10L, 0L)), 0, torch_quint8())$int_repr()
NULL
# -> quantize_per_channel <-

# -> meshgrid: 496caf98702e12950b9ddb68c3ac2985 <-
#'
#' @name torch_meshgrid
#'
#' @examples
#'
#' x = torch_tensor(c(1, 2, 3))
#' y = torch_tensor(c(4, 5, 6))
#' out = torch_meshgrid(list(x, y))
#' out
NULL
# -> meshgrid <-

# -> cartesian_prod: 86b159a5b4c055cd6c77c3e5e1215fca <-
#'
#' @name torch_cartesian_prod
#'
#' @examples
#'
#' a = c(1, 2, 3)
#' b = c(4, 5)
#' tensor_a = torch_tensor(a)
#' tensor_b = torch_tensor(b)
#' torch_cartesian_prod(list(tensor_a, tensor_b))
NULL
# -> cartesian_prod <-

# -> combinations: f79afc304b65429e99b41f6d4ec67211 <-
#'
#' @name torch_combinations
#'
#' @examples
#'
#' a = c(1, 2, 3)
#' tensor_a = torch_tensor(a)
#' torch_combinations(tensor_a)
#' torch_combinations(tensor_a, r=3)
#' torch_combinations(tensor_a, with_replacement=TRUE)
NULL
# -> combinations <-

# -> result_type: c18a90be7365222d04e6c6d98b05ccf1 <-
#'
#' @name torch_result_type
#'
#' @examples
#'
#' torch_result_type(tensor = torch_tensor(c(1, 2), dtype=torch_int()), 1.0)
NULL
# -> result_type <-

# -> can_cast: ceb9cd9a330a8fcff289af5c24c2c786 <-
#'
#' @name torch_can_cast
#'
#' @examples
#'
#' torch_can_cast(torch_double(), torch_float())
#' torch_can_cast(torch_float(), torch_int())
NULL
# -> can_cast <-

# -> promote_types: 8b3e750921514b2d4a64936cd0a71eb8 <-
#'
#' @name torch_promote_types
#'
#' @examples
#'
#' torch_promote_types(torch_int32(), torch_float32())
#' torch_promote_types(torch_uint8(), torch_long())
NULL
# -> promote_types <-

# -> bitwise_xor: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_bitwise_xor
#'
#'
NULL
# -> bitwise_xor <-

# -> addbmm: 2960d7952feaac5393ad64ea57e69ea2 <-
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
# -> addbmm <-

# -> diag: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_diag
#'
#'
NULL
# -> diag <-

# -> cross: 5cd5cf62e999579a8349e3bc1817aa7d <-
#'
#' @name torch_cross
#'
#' @examples
#'
#' a = torch_randn(c(4, 3))
#' a
#' b = torch_randn(c(4, 3))
#' b
#' torch_cross(a, b, dim=2)
#' torch_cross(a, b)
NULL
# -> cross <-

# -> triu: 9d9e5620c90c5b08ef42c88df4cecabd <-
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
# -> triu <-

# -> tril: 9476db556baac0139ee3adc08fd737c6 <-
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
# -> tril <-

# -> tril_indices: 204301311a569bc8d7a607235b6ef7a5 <-
#'
#' @name torch_tril_indices
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
NULL
# -> tril_indices <-

# -> triu_indices: c51155db9b289aafdc030310a8cb83ff <-
#'
#' @name torch_triu_indices
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
NULL
# -> triu_indices <-

# -> trace: 22523618567c5d2a75ab44934ae4dda9 <-
#'
#' @name torch_trace
#'
#' @examples
#'
#' x = torch_arange(1., 10.)$view(c(3, 3))
#' x
#' torch_trace(x)
NULL
# -> trace <-

# -> ne: c8ea98d1362a7ed6745931fd3387cd02 <-
#'
#' @name torch_ne
#'
#' @examples
#'
#' torch_ne(torch_tensor(matrix(1:4, ncol = 2, byrow=TRUE)), 
#'          torch_tensor(matrix(rep(c(1,4), each = 2), ncol = 2, byrow=TRUE)))
NULL
# -> ne <-

# -> eq: be7c586a3ad2bd934e54f2dce6b3ed27 <-
#'
#' @name torch_eq
#'
#' @examples
#'
#' torch_eq(torch_tensor(c(1,2,3,4)), torch_tensor(c(1, 3, 2, 4)))
NULL
# -> eq <-

# -> ge: bc97684018e2c69f04155c176c7ff136 <-
#'
#' @name torch_ge
#'
#' @examples
#'
#' torch_ge(torch_tensor(matrix(1:4, ncol = 2, byrow=TRUE)), 
#'          torch_tensor(matrix(c(1,1,4,4), ncol = 2, byrow=TRUE)))
NULL
# -> ge <-

# -> le: 51b6c3bd66c01f03698e01352cf8f91b <-
#'
#' @name torch_le
#'
#' @examples
#'
#' torch_le(torch_tensor(matrix(1:4, ncol = 2, byrow=TRUE)), 
#'          torch_tensor(matrix(c(1,1,4,4), ncol = 2, byrow=TRUE)))
NULL
# -> le <-

# -> gt: 90062435ff09b698ddab5f120bb2ccbb <-
#'
#' @name torch_gt
#'
#' @examples
#'
#' torch_gt(torch_tensor(matrix(1:4, ncol = 2, byrow=TRUE)), 
#'          torch_tensor(matrix(c(1,1,4,4), ncol = 2, byrow=TRUE)))
NULL
# -> gt <-

# -> lt: 67ea0b1d72c2c8ecf4c9eefe67c6617e <-
#'
#' @name torch_lt
#'
#' @examples
#'
#' torch_lt(torch_tensor(matrix(1:4, ncol = 2, byrow=TRUE)), 
#'          torch_tensor(matrix(c(1,1,4,4), ncol = 2, byrow=TRUE)))
NULL
# -> lt <-

# -> take: 2634a3085985dded8ac6c241469ad326 <-
#'
#' @name torch_take
#'
#' @examples
#'
#' src = torch_tensor(matrix(c(4,3,5,6,7,8), ncol = 3, byrow = TRUE))
#' torch_take(src, torch_tensor(c(1, 2, 5), dtype = torch_int64()))
NULL
# -> take <-

# -> index_select: b652b7dcc296fd86f3f57f0f7ecdf062 <-
#'
#' @name torch_index_select
#'
#' @examples
#'
#' x = torch_randn(c(3, 4))
#' x
#' indices = torch_tensor(c(1, 3), dtype = torch_int64())
#' torch_index_select(x, 1, indices)
#' torch_index_select(x, 2, indices)
NULL
# -> index_select <-

# -> masked_select: e70a628493edc8173a2cf1fcd186a248 <-
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
# -> masked_select <-

# -> nonzero: 6d8257748a09b7b58993c46b38f0e31a <-
#'
#' @name torch_nonzero
#'
#' @examples
#'
#' torch_nonzero(torch_tensor(c(1, 1, 1, 0, 1)))
NULL
# -> nonzero <-

# -> gather: afb4c650babd359a0e25636b444741ad <-
#'
#' @name torch_gather
#'
#' @examples
#'
#' t = torch_tensor(matrix(c(1,2,3,4), ncol = 2, byrow = TRUE))
#' torch_gather(t, 2, torch_tensor(matrix(c(1,1,2,1), ncol = 2, byrow=TRUE), dtype = torch_int64()))
NULL
# -> gather <-

# -> addcmul: 2af9b1673156c083576a65ace386d304 <-
#'
#' @name torch_addcmul
#'
#' @examples
#'
#' t = torch_randn(c(1, 3))
#' t1 = torch_randn(c(3, 1))
#' t2 = torch_randn(c(1, 3))
#' torch_addcmul(t, t1, t2, 0.1)
NULL
# -> addcmul <-

# -> addcdiv: a3fcec9931af0042b5f5fac6f812b269 <-
#'
#' @name torch_addcdiv
#'
#' @examples
#'
#' t = torch_randn(c(1, 3))
#' t1 = torch_randn(c(3, 1))
#' t2 = torch_randn(c(1, 3))
#' torch_addcdiv(t, t1, t2, 0.1)
NULL
# -> addcdiv <-

# -> lstsq: 0af50cba3dcf8ffd80ad812162fb7f33 <-
#'
#' @name torch_lstsq
#'
#' @examples
#'
#' A = torch_tensor(rbind(
#'  c(1,1,1),
#'  c(2,3,4),
#'  c(3,5,2),
#'  c(4,2,5),
#'  c(5,4,3)
#' ))
#' B = torch_tensor(rbind(
#'  c(-10, -3),
#'  c(12, 14),
#'  c(14, 12),
#'  c(16, 16),
#'  c(18, 16)
#' ))
#' out = torch_lstsq(B, A)
#' out[[1]]
NULL
# -> lstsq <-

# -> triangular_solve: 3a6185e8a4281e12dc6c7873a1484b31 <-
#'
#' @name torch_triangular_solve
#'
#' @examples
#'
#' A = torch_randn(c(2, 2))$triu()
#' A
#' b = torch_randn(c(2, 3))
#' b
#' torch_triangular_solve(b, A)
NULL
# -> triangular_solve <-

# -> symeig: 27bc25d51797de06954ef84fde11f765 <-
#'
#' @name torch_symeig
#'
#' @examples
#'
#' a = torch_randn(c(5, 5))
#' a = a + a$t()  # To make a symmetric
#' a
#' o = torch_symeig(a, eigenvectors=TRUE)
#' e = o[[1]]
#' v = o[[2]]
#' e
#' v
#' a_big = torch_randn(c(5, 2, 2))
#' a_big = a_big + a_big$transpose(-2, -1)  # To make a_big symmetric
#' o = a_big$symeig(eigenvectors=TRUE)
#' e = o[[1]]
#' v = o[[2]]
#' torch_allclose(torch_matmul(v, torch_matmul(e$diag_embed(), v$transpose(-2, -1))), a_big)
NULL
# -> symeig <-

# -> eig: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_eig
#'
#'
NULL
# -> eig <-

# -> svd: 47d9bb7bdad254caeab8884b29d7d089 <-
#'
#' @name torch_svd
#'
#' @examples
#'
#' a = torch_randn(c(5, 3))
#' a
#' out = torch_svd(a)
#' u = out[[1]]
#' s = out[[2]]
#' v = out[[3]]
#' torch_dist(a, torch_mm(torch_mm(u, torch_diag(s)), v$t()))
#' a_big = torch_randn(c(7, 5, 3))
#' out = torch_svd(a_big)
#' u = out[[1]]
#' s = out[[2]]
#' v = out[[3]]
#' torch_dist(a_big, torch_matmul(torch_matmul(u, torch_diag_embed(s)), v$transpose(-2, -1)))
NULL
# -> svd <-

# -> cholesky: fa4f9078765e51caac6f19460a4c298a <-
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
#' \dontrun{
#' a = torch_matmul(a, a$transpose(-1, -2)) + 1e-03 # make symmetric positive-definite
#' l = torch_cholesky(a)
#' z = torch_matmul(l, l$transpose(-1, -2))
#' torch_max(torch_abs(z - a)) # Max non-zero
#' }
NULL
# -> cholesky <-

# -> cholesky_solve: 12d0129f721e8467f93ffe629e8e569e <-
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
# -> cholesky_solve <-

# -> solve: df50588b379236734cc9a64acbbce5ae <-
#'
#' @name torch_solve
#'
#' @examples
#'
#' A = torch_tensor(rbind(c(6.80, -2.11,  5.66,  5.97,  8.23),
#'                       c(-6.05, -3.30,  5.36, -4.44,  1.08),
#'                       c(-0.45,  2.58, -2.70,  0.27,  9.04),
#'                       c(8.32,  2.71,  4.35,  -7.17,  2.14),
#'                       c(-9.67, -5.14, -7.26,  6.08, -6.87)))$t()
#' B = torch_tensor(rbind(c(4.02,  6.19, -8.22, -7.57, -3.03),
#'                       c(-1.56,  4.00, -8.67,  1.75,  2.86),
#'                       c(9.81, -4.09, -4.57, -8.61,  8.99)))$t()
#' out = torch_solve(B, A)
#' X = out[[1]]
#' LU = out[[2]]
#' torch_dist(B, torch_mm(A, X))
#' # Batched solver example
#' A = torch_randn(c(2, 3, 1, 4, 4))
#' B = torch_randn(c(2, 3, 1, 4, 6))
#' out = torch_solve(B, A)
#' X = out[[1]]
#' LU = out[[2]]
#' torch_dist(B, A$matmul(X))
NULL
# -> solve <-

# -> cholesky_inverse: 848d88acf5dc9debea4b899b298b870d <-
#'
#' @name torch_cholesky_inverse
#'
#' @examples
#'
#' \dontrun{
#' a = torch_randn(c(3, 3))
#' a = torch_mm(a, a$t()) + 1e-05 * torch_eye(3) # make symmetric positive definite
#' u = torch_cholesky(a)
#' a
#' torch_cholesky_inverse(u)
#' a$inverse()
#' }
NULL
# -> cholesky_inverse <-

# -> qr: cd13e92083d63cd10fa043114695d130 <-
#'
#' @name torch_qr
#'
#' @examples
#'
#' a = torch_tensor(matrix(c(12., -51, 4, 6, 167, -68, -4, 24, -41), ncol = 3, byrow = TRUE))
#' out = torch_qr(a)
#' q = out[[1]]
#' r = out[[2]]
#' torch_mm(q, r)$round()
#' torch_mm(q$t(), q)$round()
NULL
# -> qr <-

# -> geqrf: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_geqrf
#'
#'
NULL
# -> geqrf <-

# -> orgqr: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_orgqr
#'
#'
NULL
# -> orgqr <-

# -> ormqr: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_ormqr
#'
#'
NULL
# -> ormqr <-

# -> lu_solve: c0f3ad3b69367dc30938327109f883e8 <-
#'
#' @name torch_lu_solve
#'
#' @examples
#' A = torch_randn(c(2, 3, 3))
#' b = torch_randn(c(2, 3, 1))
#' out = torch_lu(A)
#' x = torch_lu_solve(b, out[[1]], out[[2]])
#' torch_norm(torch_bmm(A, x) - b)
NULL
# -> lu_solve <-

# -> multinomial: 21dad4638e2dbe0c36cd0cd3e0e1e12d <-
#'
#' @name torch_multinomial
#'
#' @examples
#'
#' weights = torch_tensor(c(0, 10, 3, 0), dtype=torch_float()) # create a tensor of weights
#' torch_multinomial(weights, 2)
#' torch_multinomial(weights, 4, replacement=TRUE)
NULL
# -> multinomial <-

# -> lgamma: f7fea324140b7b370280b904c2c2a6ae <-
#'
#' @name torch_lgamma
#'
#' @examples
#'
#' a = torch_arange(0.5, 2, 0.5)
#' torch_lgamma(a)
NULL
# -> lgamma <-

# -> digamma: ef012b03c92ee9f005d3b5a392340dc0 <-
#'
#' @name torch_digamma
#'
#' @examples
#'
#' a = torch_tensor(c(1, 0.5))
#' torch_digamma(a)
NULL
# -> digamma <-

# -> polygamma: f6adb11d5a3f6e72157a32551e4584d8 <-
#'
#' @name torch_polygamma
#'
#' @examples
#' \dontrun{
#' a = torch_tensor(c(1, 0.5))
#' torch_polygamma(1, a)
#' }
NULL
# -> polygamma <-

# -> erfinv: 4e5124ffa5fbdb4aa818c64496d49b51 <-
#'
#' @name torch_erfinv
#'
#' @examples
#'
#' torch_erfinv(torch_tensor(c(0, 0.5, -1.)))
NULL
# -> erfinv <-

# -> sign: 412fe6a4bb2dfe9370e746b0f231267a <-
#'
#' @name torch_sign
#'
#' @examples
#'
#' a = torch_tensor(c(0.7, -1.2, 0., 2.3))
#' a
#' torch_sign(a)
NULL
# -> sign <-

# -> dist: 6ecd50cefe7774770927770836f1db55 <-
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
# -> dist <-

# -> atan2: 079d8dfe03eb6076301dccc1f386d800 <-
#'
#' @name torch_atan2
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_atan2(a, torch_randn(c(4)))
NULL
# -> atan2 <-

# -> lerp: f0f3a0218d7e6d89ee7397124f3ebd0f <-
#'
#' @name torch_lerp
#'
#' @examples
#'
#' start = torch_arange(1., 5.)
#' end = torch_empty(4)$fill_(10)
#' start
#' end
#' torch_lerp(start, end, 0.5)
#' torch_lerp(start, end, torch_full_like(start, 0.5))
NULL
# -> lerp <-

# -> histc: d3ac23440a13017904435b373a289ebc <-
#'
#' @name torch_histc
#'
#' @examples
#'
#' torch_histc(torch_tensor(c(1., 2, 1)), bins=4, min=0, max=3)
NULL
# -> histc <-

# -> fmod: 48f0420c8701cca2f79554e7dfe9f5e4 <-
#'
#' @name torch_fmod
#'
#' @examples
#'
#' torch_fmod(torch_tensor(c(-3., -2, -1, 1, 2, 3)), 2)
#' torch_fmod(torch_tensor(c(1., 2, 3, 4, 5)), 1.5)
NULL
# -> fmod <-

# -> remainder: b20c6555afa7d27c1140ff1f5b337bf4 <-
#'
#' @name torch_remainder
#'
#' @examples
#'
#' torch_remainder(torch_tensor(c(-3., -2, -1, 1, 2, 3)), 2)
#' torch_remainder(torch_tensor(c(1., 2, 3, 4, 5)), 1.5)
NULL
# -> remainder <-

# -> sort: 9f1fc598fbb5f20f000878002ee1299b <-
#'
#' @name torch_sort
#'
#' @examples
#'
#' x = torch_randn(c(3, 4))
#' out = torch_sort(x)
#' out
#' out = torch_sort(x, 1)
#' out
NULL
# -> sort <-

# -> argsort: c5596fbc697d7819566e3002ae6e4647 <-
#'
#' @name torch_argsort
#'
#' @examples
#'
#' a = torch_randn(c(4, 4))
#' a
#' torch_argsort(a, dim=1)
NULL
# -> argsort <-

# -> topk: c4a40d9553a19596080b2fe029ce3d76 <-
#'
#' @name torch_topk
#'
#' @examples
#'
#' x = torch_arange(1., 6.)
#' x
#' torch_topk(x, 3)
NULL
# -> topk <-

# -> renorm: a7b54455c60ff955000945fb62311fa6 <-
#'
#' @name torch_renorm
#'
#' @examples
#' x = torch_ones(c(3, 3))
#' x[2,]$fill_(2)
#' x[3,]$fill_(3)
#' x
#' torch_renorm(x, 1, 1, 5)
NULL
# -> renorm <-

# -> equal: 7c3ca1e7965de36729bbb40352c33d1b <-
#'
#' @name torch_equal
#'
#' @examples
#'
#' torch_equal(torch_tensor(c(1, 2)), torch_tensor(c(1, 2)))
NULL
# -> equal <-

# -> normal: d00d2407a603ac28ea06472e4f0b71de <-
#'
#' @name torch_normal
#'
#' @examples
#'
#' \dontrun{
#' torch_normal(mean=0, std=torch_arange(1, 0, -0.1))
#'
#'
#' torch_normal(mean=0.5, std=torch_arange(1., 6.))
#'
#'
#' torch_normal(mean=torch_arange(1., 6.))
#'
#'
#' torch_normal(2, 3, size=list(1, 4))
#' }
NULL
# -> normal <-

# -> isfinite: 2e310d0d6bec1a2aa45b6477e93861b7 <-
#'
#' @name torch_isfinite
#'
#' @examples
#'
#' torch_isfinite(torch_tensor(c(1, Inf, 2, -Inf, NaN)))
NULL
# -> isfinite <-

# -> logical_and: 2fd62dde47e072501a4e652b24862d03 <-
#'
#' @name torch_logical_and
#'
#' @examples
#'
#' torch_logical_and(torch_tensor(c(TRUE, FALSE, TRUE)), torch_tensor(c(TRUE, FALSE, FALSE)))
#' a = torch_tensor(c(0, 1, 10, 0), dtype=torch_int8())
#' b = torch_tensor(c(4, 0, 1, 0), dtype=torch_int8())
#' torch_logical_and(a, b)
#' \dontrun{
#' torch_logical_and(a, b, out=torch_empty(4, dtype=torch_bool()))
#' }
NULL
# -> logical_and <-

# -> logical_or: b02dcb9041f32e5506dd4ea911f523e2 <-
#'
#' @name torch_logical_or
#'
#' @examples
#'
#' torch_logical_or(torch_tensor(c(TRUE, FALSE, TRUE)), torch_tensor(c(TRUE, FALSE, FALSE)))
#' a = torch_tensor(c(0, 1, 10, 0), dtype=torch_int8())
#' b = torch_tensor(c(4, 0, 1, 0), dtype=torch_int8())
#' torch_logical_or(a, b)
#' \dontrun{
#' torch_logical_or(a$double(), b$double())
#' torch_logical_or(a$double(), b)
#' torch_logical_or(a, b, out=torch_empty(4, dtype=torch_bool()))
#' }
NULL
# -> logical_or <-

# -> cummax: c6caab76ec2e2a82766108186c3e736d <-
#'
#' @name torch_cummax
#'
#' @examples
#'
#' a = torch_randn(c(10))
#' a
#' torch_cummax(a, dim=1)
NULL
# -> cummax <-

# -> cummin: 9c7c3817d592a264549692ecd0b8440e <-
#'
#' @name torch_cummin
#'
#' @examples
#'
#' a = torch_randn(c(10))
#' a
#' torch_cummin(a, dim=1)
NULL
# -> cummin <-

# -> floor_divide: c913397c4c025ed2e753191375508813 <-
#'
#' @name torch_floor_divide
#'
#' @examples
#'
#' a = torch_tensor(c(4.0, 3.0))
#' b = torch_tensor(c(2.0, 2.0))
#' torch_floor_divide(a, b)
#' torch_floor_divide(a, 1.4)
NULL
# -> floor_divide <-

# -> is_complex: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_is_complex
#'
#'
NULL
# -> is_complex <-

# -> square: cf1b3195377f1b4e052e22932d61b186 <-
#'
#' @name torch_square
#'
#' @examples
#'
#' a = torch_randn(c(4))
#' a
#' torch_square(a)
NULL
# -> square <-

# -> true_divide: 7bcbf6c439736cf9eb725cec8d1ca73c <-
#'
#' @name torch_true_divide
#'
#' @examples
#'
#' dividend = torch_tensor(c(5, 3), dtype=torch_int())
#' divisor = torch_tensor(c(3, 2), dtype=torch_int())
#' torch_true_divide(dividend, divisor)
#' torch_true_divide(dividend, 2)
NULL
# -> true_divide <-

# -> poisson: 4ab231954488da8fd133ce813da9d696 <-
#'
#' @name torch_poisson
#'
#' @examples
#'
#' rates = torch_rand(c(4, 4)) * 5  # rate parameter between 0 and 5
#' torch_poisson(rates)
NULL
# -> poisson <-

# -> bitwise_and: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_bitwise_and
#'
#'
NULL
# -> bitwise_and <-

# -> bitwise_or: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_bitwise_or
#'
#'
NULL
# -> bitwise_or <-

# -> isinf: 7c5614e21bd63db036d261ef73b2f1f2 <-
#'
#' @name torch_isinf
#'
#' @examples
#'
#' torch_isinf(torch_tensor(c(1, Inf, 2, -Inf, NaN)))
NULL
# -> isinf <-