

# -> view_as_real: a3f20b05fb24decaed1495b2cc09360b <-
#'
#' @name torch_view_as_real
#'
#' @examples
#'
#' if (FALSE) {
#' x <- torch_randn(4, dtype=torch_cfloat())
#' x
#' torch_view_as_real(x)
#' }
NULL
# -> view_as_real <-

# -> view_as_complex: cf6bf0120d9c60c6efb5077f4fddb53c <-
#'
#' @name torch_view_as_complex
#'
#' @examples
#' if (FALSE) {
#' x=torch_randn(c(4, 2))
#' x
#' torch_view_as_complex(x)
#' }
NULL
# -> view_as_complex <-

# -> sgn: 2060765e46e5dad864e241c9b6be30da <-
#'
#' @name torch_sgn
#'
#' @examples
#' if (FALSE) {
#' x <- torch_tensor(c(3+4j, 7-24j, 0, 1+2j))
#' x$sgn()
#' torch_sgn(x)
#' }
NULL
# -> sgn <-

# -> arccos: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_arccos
#'
#'
NULL
# -> arccos <-

# -> acosh: 1d78f75c41fbecf646f9db538771cebc <-
#'
#' @name torch_acosh
#'
#' @examples
#'
#' a <- torch_randn(c(4))$uniform_(1, 2)
#' a
#' torch_acosh(a)
NULL
# -> acosh <-

# -> arccosh: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_arccosh
#'
#'
NULL
# -> arccosh <-

# -> asinh: ca41bf50229587c3c5374be2962f8d75 <-
#'
#' @name torch_asinh
#'
#' @examples
#'
#' a <- torch_randn(c(4))
#' a
#' torch_asinh(a)
NULL
# -> asinh <-

# -> arcsinh: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_arcsinh
#'
#'
NULL
# -> arcsinh <-

# -> atanh: 7c32bcb3b95672b3ee919e1284511c6d <-
#'
#' @name torch_atanh
#'
#' @examples
#'
#' a = torch_randn(c(4))$uniform_(-1, 1)
#' a
#' torch_atanh(a)
NULL
# -> atanh <-

# -> arctanh: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_arctanh
#'
#'
NULL
# -> arctanh <-

# -> arcsin: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_arcsin
#'
#'
NULL
# -> arcsin <-

# -> arctan: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_arctan
#'
#'
NULL
# -> arctan <-

# -> atleast_1d: 6aeb818b1f495c13b606f8f35b9cbd04 <-
#'
#' @name torch_atleast_1d
#'
#' @examples
#'
#' x <- torch_randn(c(2))
#' x
#' torch_atleast_1d(x)
#' x <- torch_tensor(1.)
#' x
#' torch_atleast_1d(x)
#' x <- torch_tensor(0.5)
#' y <- torch_tensor(1.)
#' torch_atleast_1d(list(x,y))
NULL
# -> atleast_1d <-

# -> atleast_2d: 083980db60eb0e2ef87b537b4351da06 <-
#'
#' @name torch_atleast_2d
#'
#' @examples
#'
#' x <- torch_tensor(1.)
#' x
#' torch_atleast_2d(x)
#' x <- torch_randn(c(2,2))
#' x
#' torch_atleast_2d(x)
#' x <- torch_tensor(0.5)
#' y <- torch_tensor(1.)
#' torch_atleast_2d(list(x,y))
NULL
# -> atleast_2d <-

# -> atleast_3d: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_atleast_3d
#'
#'
NULL
# -> atleast_3d <-

# -> block_diag: c16795e6691e3e921cc9eeb615d52c7c <-
#'
#' @name torch_block_diag
#'
#' @examples
#'
#' A <- torch_tensor(rbind(c(0, 1), c(1, 0)))
#' B <- torch_tensor(rbind(c(3, 4, 5), c(6, 7, 8)))
#' C <- torch_tensor(7)
#' D <- torch_tensor(c(1, 2, 3))
#' E <- torch_tensor(rbind(4, 5, 6))
#' torch_block_diag(list(A, B, C, D, E))
NULL
# -> block_diag <-

# -> unsafe_chunk: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_unsafe_chunk
#'
#'
NULL
# -> unsafe_chunk <-

# -> clip: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_clip
#'
#'
NULL
# -> clip <-

# -> complex: e4d3843d2a8f605b807c3202d6f293b2 <-
#'
#' @name torch_complex
#'
#' @examples
#'
#' real <- torch_tensor(c(1, 2), dtype=torch_float32())
#' imag <- torch_tensor(c(3, 4), dtype=torch_float32())
#' z <- torch_complex(real, imag)
#' z
#' z$dtype
NULL
# -> complex <-

# -> polar: 468876b978d148a914febded0324c5ff <-
#'
#' @name torch_polar
#'
#' @examples
#'
#' abs <- torch_tensor(c(1, 2), dtype=torch_float64())
#' angle <- torch_tensor(c(pi / 2, 5 * pi / 4), dtype=torch_float64())
#' z <- torch_polar(abs, angle)
#' z
NULL
# -> polar <-

# -> count_nonzero: b446d63fd33a0b5a01d71b0ef2eaab45 <-
#'
#' @name torch_count_nonzero
#'
#' @examples
#'
#' x <- torch_zeros(3,3)
#' x[torch_randn(3,3) > 0.5] = 1
#' x
#' torch_count_nonzero(x)
#' torch_count_nonzero(x, dim=0)
NULL
# -> count_nonzero <-

# -> divide: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_divide
#'
#'
NULL
# -> divide <-

# -> vdot: fd2819b470f5d0bdd82a72eba1ec4714 <-
#'
#' @name torch_vdot
#'
#' @examples
#'
#' torch_vdot(torch_tensor(c(2, 3)), torch_tensor(c(2, 1)))
#' if (FALSE) {
#' a <- torch_tensor(list(1 +2j, 3 - 1i))
#' b <- torch_tensor(list(2 +1j, 4 - 0i))
#' torch_vdot(a, b)
#' torch_vdot(b, a)
#' }
NULL
# -> vdot <-

# -> exp2: 7509fd2dd8ce681440c02d1bfc95b66c <-
#'
#' @name torch_exp2
#'
#' @examples
#'
#' torch_exp2(torch_tensor(c(0, log2(2.), 3, 4)))
NULL
# -> exp2 <-

# -> gcd: 7f06533263345702199f039959e63836 <-
#'
#' @name torch_gcd
#'
#' @examples
#'
#' if (torch::cuda_is_available()) {
#' a <- torch_tensor(c(5, 10, 15))
#' b <- torch_tensor(c(3, 4, 5))
#' torch_gcd(a, b)
#' c <- torch_tensor(c(3))
#' torch_gcd(a, c)
#' }
NULL
# -> gcd <-

# -> lcm: d706df34bbb3a8f6404f9a22f776b470 <-
#'
#' @name torch_lcm
#'
#' @examples
#'
#' if (torch::cuda_is_available()) {
#' a <- torch_tensor(c(5, 10, 15))
#' b <- torch_tensor(c(3, 4, 5))
#' torch_lcm(a, b)
#' c <- torch_tensor(c(3))
#' torch_lcm(a, c)
#' }
NULL
# -> lcm <-

# -> kaiser_window: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_kaiser_window
#'
#'
NULL
# -> kaiser_window <-

# -> isclose: d2b0783ddc0e65e8a4556468ec25d8d1 <-
#'
#' @name torch_isclose
#'
#' @examples
#'
#' torch_isclose(torch_tensor(c(1., 2, 3)), torch_tensor(c(1 + 1e-10, 3, 4)))
#' torch_isclose(torch_tensor(c(Inf, 4)), torch_tensor(c(Inf, 6)), rtol=.5)
NULL
# -> isclose <-

# -> isreal: 848e6b56d0e94c525a8077345511f58d <-
#'
#' @name torch_isreal
#'
#' @examples
#' if (FALSE) {
#' torch_isreal(torch_tensor(c(1, 1+1j, 2+0j)))
#' }
NULL
# -> isreal <-

# -> is_nonzero: 04f3fe53d2f6923e62d6b3f98d431851 <-
#'
#' @name torch_is_nonzero
#'
#' @examples
#'
#' torch_is_nonzero(torch_tensor(c(0.)))
#' torch_is_nonzero(torch_tensor(c(1.5)))
#' torch_is_nonzero(torch_tensor(c(FALSE)))
#' torch_is_nonzero(torch_tensor(c(3)))
#' if (FALSE) {
#' torch_is_nonzero(torch_tensor(c(1, 3, 5)))
#' torch_is_nonzero(torch_tensor(c()))
#' }
NULL
# -> is_nonzero <-

# -> logaddexp: d9de4be16d81aa701335694e1b26c27d <-
#'
#' @name torch_logaddexp
#'
#' @examples
#'
#' torch_logaddexp(torch_tensor(c(-1.0)), torch_tensor(c(-1.0, -2, -3)))
#' torch_logaddexp(torch_tensor(c(-100.0, -200, -300)), torch_tensor(c(-1.0, -2, -3)))
#' torch_logaddexp(torch_tensor(c(1.0, 2000, 30000)), torch_tensor(c(-1.0, -2, -3)))
NULL
# -> logaddexp <-

# -> logaddexp2: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_logaddexp2
#'
#'
NULL
# -> logaddexp2 <-

# -> logcumsumexp: a0097c23e5c6fca885055b408298b28a <-
#'
#' @name torch_logcumsumexp
#'
#' @examples
#'
#' a <- torch_randn(c(10))
#' torch_logcumsumexp(a, dim=1)
NULL
# -> logcumsumexp <-

# -> matrix_exp: 4f7ee042420b5500c0fd5cb1f2238b1b <-
#'
#' @name torch_matrix_exp
#'
#' @examples
#'
#' a <- torch_randn(c(2, 2, 2))
#' a[1, , ] <- torch_eye(2, 2)
#' a[2, , ] <- 2 * torch_eye(2, 2)
#' a
#' torch_matrix_exp(a)
#' 
#' x <- torch_tensor(rbind(c(0, pi/3), c(-pi/3, 0)))
#' x$matrix_exp() # should be [[cos(pi/3), sin(pi/3)], [-sin(pi/3), cos(pi/3)]]
NULL
# -> matrix_exp <-

# -> amax: 9829bb50baa6a30166e72d47828b24c3 <-
#'
#' @name torch_amax
#'
#' @examples
#'
#' a <- torch_randn(c(4, 4))
#' a
#' torch_amax(a, 1)
NULL
# -> amax <-

# -> amin: f617e1319f007cc62b71bc1566370dac <-
#'
#' @name torch_amin
#'
#' @examples
#'
#' a <- torch_randn(c(4, 4))
#' a
#' torch_amin(a, 1)
NULL
# -> amin <-

# -> multiply: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_multiply
#'
#'
NULL
# -> multiply <-

# -> movedim: 97e5fa29a75fe27fe6b650ba89da2650 <-
#'
#' @name torch_movedim
#'
#' @examples
#'
#' t <- torch_randn(c(3,2,1))
#' t
#' torch_movedim(t, 2, 1)$shape
#' torch_movedim(t, 2, 1)
#' torch_movedim(t, c(2, 3), c(1, 2))$shape
#' torch_movedim(t, c(2, 3), c(1, 2))
NULL
# -> movedim <-

# -> channel_shuffle: 2d94efbd431f65a9a374ed335399bb5c <-
#'
#' @name torch_channel_shuffle
#'
#' @examples
#'
#' input = torch_randn(c(1, 4, 2, 2))
#' print(input)
#' output = torch_nn$functional$channel_shuffle(input, 2)
#' print(output)
NULL
# -> channel_shuffle <-

# -> rad2deg: 348b68fbd67c8cddb92abfd326fc26c7 <-
#'
#' @name torch_rad2deg
#'
#' @examples
#'
#' a = torch_tensor(c([3.142, -3.142], [6.283, -6.283], [1.570, -1.570]))
#' torch_rad2deg(a)
NULL
# -> rad2deg <-

# -> deg2rad: e8c585843d13b66762128e3f30264275 <-
#'
#' @name torch_deg2rad
#'
#' @examples
#'
#' a = torch_tensor(c([180.0, -180.0], [360.0, -360.0], [90.0, -90.0]))
#' torch_deg2rad(a)
NULL
# -> deg2rad <-

# -> negative: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_negative
#'
#'
NULL
# -> negative <-

# -> logit: bc3da194315910774b7b315b58fdda61 <-
#'
#' @name torch_logit
#'
#' @examples
#'
#' a = torch_rand(5)
#' a
#' torch_logit(a, eps=1e-6)
NULL
# -> logit <-

# -> unsafe_split: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_unsafe_split
#'
#'
NULL
# -> unsafe_split <-

# -> hstack: 0901dedbfaf0f8d494ff57b1384d28c5 <-
#'
#' @name torch_hstack
#'
#' @examples
#'
#' a = torch_tensor(c(1, 2, 3))
#' b = torch_tensor(c(4, 5, 6))
#' torch_hstack(list(a,b))
#' a = torch_tensor(c([1],[2],[3]))
#' b = torch_tensor(c([4],[5],[6]))
#' torch_hstack(list(a,b))
NULL
# -> hstack <-

# -> vstack: c321a7157cffb50ef7463b3c85a50573 <-
#'
#' @name torch_vstack
#'
#' @examples
#'
#' a = torch_tensor(c(1, 2, 3))
#' b = torch_tensor(c(4, 5, 6))
#' torch_vstack(list(a,b))
#' a = torch_tensor(c([1],[2],[3]))
#' b = torch_tensor(c([4],[5],[6]))
#' torch_vstack(list(a,b))
NULL
# -> vstack <-

# -> dstack: 14253e36bf647f8b97538f9e23f17aec <-
#'
#' @name torch_dstack
#'
#' @examples
#'
#' a = torch_tensor(c(1, 2, 3))
#' b = torch_tensor(c(4, 5, 6))
#' torch_dstack(list(a,b))
#' a = torch_tensor(c([1],[2],[3]))
#' b = torch_tensor(c([4],[5],[6]))
#' torch_dstack(list(a,b))
NULL
# -> dstack <-

# -> istft: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_istft
#'
#'
NULL
# -> istft <-

# -> nansum: 7befe938473b2951d888b8ca87a4c654 <-
#'
#' @name torch_nansum
#'
#' @examples
#'
#' a = torch_tensor(c(1., 2., NaN, 4.))
#' torch_nansum(a)
#'
#'
#' torch_nansum(torch_tensor([1., float("nan")]))
#' a = torch_tensor(c([1, 2), [3., float("nan")]])
#' torch_nansum(a)
#' torch_nansum(a, dim=0)
#' torch_nansum(a, dim=1)
NULL
# -> nansum <-

# -> fliplr: 5e8b3087ecf739091e371487701bac30 <-
#'
#' @name torch_fliplr
#'
#' @examples
#'
#' x = torch_arange(4)$view(2, 2)
#' x
#' torch_fliplr(x)
NULL
# -> fliplr <-

# -> flipud: 1d8ca3c9575b473971e177ee1233fd6c <-
#'
#' @name torch_flipud
#'
#' @examples
#'
#' x = torch_arange(4)$view(2, 2)
#' x
#' torch_flipud(x)
NULL
# -> flipud <-

# -> fix: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_fix
#'
#'
NULL
# -> fix <-

# -> vander: 2f31ff42221a9480b1a1a395f5662b3b <-
#'
#' @name torch_vander
#'
#' @examples
#'
#' x = torch_tensor(c(1, 2, 3, 5))
#' torch_vander(x)
#' torch_vander(x, N=3)
#' torch_vander(x, N=3, increasing=TRUE)
NULL
# -> vander <-

# -> clone: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_clone
#'
#'
NULL
# -> clone <-

# -> sub: 80099525a58437bf356173eaac720748 <-
#'
#' @name torch_sub
#'
#' @examples
#'
#' a = torch_tensor(list(1, 2))
#' b = torch_tensor(list(0, 1))
#' torch_sub(a, b, alpha=2)
NULL
# -> sub <-

# -> subtract: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_subtract
#'
#'
NULL
# -> subtract <-

# -> heaviside: 0329f28ac035ceccc6313cf96ca2792e <-
#'
#' @name torch_heaviside
#'
#' @examples
#'
#' input = torch_tensor(c(-1.5, 0, 2.0))
#' values = torch_tensor(c(0.5))
#' torch_heaviside(input, values)
#' values = torch_tensor(c(1.2, -2.0, 3.5))
#' torch_heaviside(input, values)
NULL
# -> heaviside <-

# -> dequantize: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_dequantize
#'
#'
NULL
# -> dequantize <-

# -> not_equal: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_not_equal
#'
#'
NULL
# -> not_equal <-

# -> greater_equal: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_greater_equal
#'
#'
NULL
# -> greater_equal <-

# -> less_equal: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_less_equal
#'
#'
NULL
# -> less_equal <-

# -> greater: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_greater
#'
#'
NULL
# -> greater <-

# -> less: 94b4710518b0b3d3bf08c75dda217258 <-
#'
#' @name torch_less
#'
#'
NULL
# -> less <-

# -> i0: 494ce1fc8c65c701da18fc37cd10572e <-
#'
#' @name torch_i0
#'
#' @examples
#'
#' torch_i0list(torch_arange(5, dtype=torch_float32()))
NULL
# -> i0 <-

# -> signbit: 0815cbe85087de269f58bc160117db27 <-
#'
#' @name torch_signbit
#'
#' @examples
#'
#' a = torch_tensor(c(0.7, -1.2, 0., 2.3))
#' torch_signbit(a)
NULL
# -> signbit <-

# -> hypot: c986a2125e5a10323c693769e6697502 <-
#'
#' @name torch_hypot
#'
#' @examples
#'
#' a = torch_hypot(torch_tensor(c(4.0)), torch_tensor(c(3.0, 4.0, 5.0)))
NULL
# -> hypot <-

# -> nextafter: aaaebf9058fb1a2d034d3ad93776ddb7 <-
#'
#' @name torch_nextafter
#'
#' @examples
#'
#' eps = torch_finfo(torch_float32)$eps
#' torch_nextafter(torch_Tensor(c(1, 2)), torch_Tensor(c(2, 1))) == torch_Tensor([eps + 1, 2 - eps])
NULL
# -> nextafter <-

# -> maximum: 4dc2812538ed437ee5aa4c46972582a5 <-
#'
#' @name torch_maximum
#'
#' @examples
#'
#' a = torch_tensor(list(1, 2, -1))
#' b = torch_tensor(list(3, 0, 4))
#' torch_maximum(a, b)
NULL
# -> maximum <-

# -> minimum: a1cf2d4621aca9402f4f534ce4829f65 <-
#'
#' @name torch_minimum
#'
#' @examples
#'
#' a = torch_tensor(list(1, 2, -1))
#' b = torch_tensor(list(3, 0, 4))
#' torch_minimum(a, b)
NULL
# -> minimum <-

# -> quantile: fdc2d4267cffea04de1a6fc2bf94acaa <-
#'
#' @name torch_quantile
#'
#' @examples
#'
#' a = torch_randn(c(1, 3))
#' a
#' q = torch_tensor(c(0, 0.5, 1))
#' torch_quantile(a, q)
#'
#'
#' a = torch_randn(c(2, 3))
#' a
#' q = torch_tensor(c(0.25, 0.5, 0.75))
#' torch_quantile(a, q, dim=1, keepdim=TRUE)
#' torch_quantile(a, q, dim=1, keepdim=TRUE)$shape
NULL
# -> quantile <-

# -> nanquantile: a53a84759ce54ded50c75bc7e761c7af <-
#'
#' @name torch_nanquantile
#'
#' @examples
#'
#' t = torch_tensor(c(NaN, 1, 2))
#' t$quantile(0.5)
#' t$nanquantile(0.5)
#' t = torch_tensor(c([NaN, NaN], [1, 2]))
#' t
#' t$nanquantile(0.5, dim=0)
#' t$nanquantile(0.5, dim=1)
NULL
# -> nanquantile <-

# -> bucketize: e49bfae68983a49dea9c55fb17cbdf81 <-
#'
#' @name torch_bucketize
#'
#' @examples
#'
#' boundaries = torch_tensor(c(1, 3, 5, 7, 9))
#' boundaries
#' v = torch_tensor(c([3, 6, 9], [3, 6, 9]))
#' v
#' torch_bucketize(v, boundaries)
#' torch_bucketize(v, boundaries, right=TRUE)
NULL
# -> bucketize <-

# -> searchsorted: 89ce755371fb059e9b773afa15e8a1b5 <-
#'
#' @name torch_searchsorted
#'
#' @examples
#'
#' sorted_sequence = torch_tensor(c([1, 3, 5, 7, 9], [2, 4, 6, 8, 10]))
#' sorted_sequence
#' values = torch_tensor(c([3, 6, 9], [3, 6, 9]))
#' values
#' torch_searchsorted(sorted_sequence, values)
#' torch_searchsorted(sorted_sequence, values, right=TRUE)
#' sorted_sequence_1d = torch_tensor(c(1, 3, 5, 7, 9))
#' sorted_sequence_1d
#' torch_searchsorted(sorted_sequence_1d, values)
NULL
# -> searchsorted <-

# -> isposinf: 61a61ec60d613dcfafa4d2339360fdf3 <-
#'
#' @name torch_isposinf
#'
#' @examples
#'
#' a = torch_tensor(c(-Inf, Inf, 1.2))
#' torch_isposinf(a)
NULL
# -> isposinf <-

# -> isneginf: 82af6b00c7afcf8419f2d6d1e6ea9177 <-
#'
#' @name torch_isneginf
#'
#' @examples
#'
#' a = torch_tensor(c(-Inf, Inf, 1.2))
#' torch_isneginf(a)
NULL
# -> isneginf <-

# -> outer: 5ab71a6034a64ef40a9755ac4962d696 <-
#'
#' @name torch_outer
#'
#' @examples
#'
#' v1 = torch_arange(1., 5.)
#' v2 = torch_arange(1., 4.)
#' torch_outer(v1, v2)
NULL
# -> outer <-