test_that("result_type", {
  x <- torch_result_type(
    tensor1 = torch_tensor(c(1, 2), dtype = torch_int()),
    tensor2 = 1
  )

  expect_true(x == torch_float())

  x <- torch_result_type(
    tensor1 = torch_tensor(c(1, 2), dtype = torch_int()),
    tensor2 = torch_tensor(1:2)
  )

  expect_true(x == torch_long())

  x <- torch_result_type(
    tensor1 = 1,
    tensor2 = torch_tensor(1:2)
  )

  expect_true(x == torch_float())

  x <- torch_result_type(
    tensor1 = 1,
    tensor2 = 2L
  )

  expect_true(x == torch_float())
})

test_that("torch_multi_margin_loss", {
  x <- torch_randn(3, 2)
  y <- torch_tensor(c(1, 2, 3), dtype = torch_long())

  expect_error(torch_multi_margin_loss(x, y))

  x <- torch_randn(3, 3)
  expect_tensor(torch_multi_margin_loss(x, y))

  y <- torch_tensor(c(0, 1, 2))
  expect_error(torch_multi_margin_loss(x, y))
})

test_that("torch_topk", {
  x <- torch_arange(1, 15)$view(c(5, 3))
  expect_equal_to_r(
    torch_topk(x, 2)[[2]],
    matrix(c(3, 2), nrow = 5, ncol = 2, byrow = TRUE)
  )

  expect_equal_to_r(
    x$topk(2)[[2]],
    matrix(c(3, 2), nrow = 5, ncol = 2, byrow = TRUE)
  )
})

test_that("torch_narrow", {
  x <- torch_tensor(matrix(1:9, ncol = 3, byrow = TRUE))
  expect_equal_to_tensor(torch_narrow(x, 1, 1, 2), x[1:2, ])
  expect_equal_to_tensor(x$narrow(1, 1, 2), x[1:2, ])
  expect_equal_to_tensor(x$narrow_copy(1, 1, 2), x[1:2, ])
})

test_that("atleast_1d", {
  x <- torch_randn(2)
  expect_equal(torch_atleast_1d(x)$ndim, 1)
  y <- torch_scalar_tensor(1)
  expect_equal(y$ndim, 0)
  expect_equal(torch_atleast_1d(y)$ndim, 1)

  z <- torch_atleast_1d(list(x, y, torch_randn(2, 2)))
  expect_equal(z[[1]]$ndim, 1)
  expect_equal(z[[2]]$ndim, 1)
  expect_equal(z[[3]]$ndim, 2)
})

test_that("atleast_2d", {
  x <- torch_randn(2)
  expect_equal(torch_atleast_2d(x)$ndim, 2)
  y <- torch_scalar_tensor(1)
  expect_equal(y$ndim, 0)
  expect_equal(torch_atleast_2d(y)$ndim, 2)

  z <- torch_atleast_2d(list(x, y, torch_randn(2, 2, 2)))
  expect_equal(z[[1]]$ndim, 2)
  expect_equal(z[[2]]$ndim, 2)
  expect_equal(z[[3]]$ndim, 3)
})

test_that("atleast_3d", {
  x <- torch_randn(2)
  expect_equal(torch_atleast_3d(x)$ndim, 3)
  y <- torch_scalar_tensor(1)
  expect_equal(y$ndim, 0)
  expect_equal(torch_atleast_3d(y)$ndim, 3)

  z <- torch_atleast_3d(list(x, y, torch_randn(2, 2, 2, 2)))
  expect_equal(z[[1]]$ndim, 3)
  expect_equal(z[[2]]$ndim, 3)
  expect_equal(z[[3]]$ndim, 4)
})

test_that("kaiser_window", {
  expect_tensor(torch_kaiser_window(10, TRUE, beta = 12))
  expect_tensor(torch_kaiser_window(10, TRUE))
  expect_tensor(torch_kaiser_window(10, FALSE))

  expect_tensor(torch_kaiser_window(10, TRUE, dtype = torch_float64()))

  x <- torch_kaiser_window(10, TRUE, dtype = torch_float64())
  expect_true(x$dtype == torch_float64())

  x <- torch_kaiser_window(10, TRUE, layout = torch_strided())
  expect_tensor(x)

  x <- torch_kaiser_window(10, TRUE, requires_grad = TRUE)
  expect_true(x$requires_grad)
})

test_that("vander", {
  x <- torch_tensor(c(1, 2, 3, 5))
  expect_tensor(torch_vander(x))

  y <- torch_vander(x, N = 3)
  expect_tensor(y)
  expect_equal(y$size(2), 3)

  y <- torch_vander(x, N = 3, increasing = TRUE)
  expect_equal_to_r(y[4, 3], 25)
})

test_that("movedim", {
  x <- torch_randn(3, 2, 1)
  expect_tensor_shape(torch_movedim(x, 1, 2), c(2, 3, 1))
  expect_tensor_shape(torch_movedim(x, c(1, 2), c(2, 3)), c(1, 3, 2))
})

test_that("norm", {
  x <- torch_rand(2, 3)
  expect_tensor(torch_norm(x))
  expect_tensor(torch_norm(x, p = 2))
  expect_tensor(torch_norm(x, p = 2, dtype = torch_float64()))
  expect_tensor_shape(torch_norm(x, dim = 1), 3)
  expect_tensor_shape(torch_norm(x, dim = 2), 2)
  expect_tensor_shape(torch_norm(x, dim = 2, dtype = torch_float64()), 2)

  x <- torch_rand(2, 3, names = c("W", "H"))
  expect_error(
    torch_norm(x, dim = "W"),
    regexp = "not yet supported with named tensors"
  )
  expect_error(
    torch_norm(x, dim = "H"),
    regexp = "not yet supported with named tensors"
  )

  x <- torch_rand(2, 3)
  expect_tensor(x$norm())
  expect_tensor(x$norm(p = 2))
  expect_tensor(x$norm(p = 2, dtype = torch_float64()))
  expect_tensor_shape(torch_norm(x, dim = 1), 3)
  expect_tensor_shape(torch_norm(x, dim = 2), 2)
  expect_tensor_shape(torch_norm(x, dim = 2, dtype = torch_float64()), 2)
})

test_that("hann_window", {
  expect_error(
    torch_hann_window(NULL),
    class = "value_error"
  )

  expect_tensor_shape(torch_hann_window(window_length = 10), 10)
})

test_that("stft", {
  x <- torch_stft(
    input = torch::torch_ones(3000),
    n_fft = 400,
    center = FALSE,
    onesided = TRUE
  )

  expect_tensor_shape(x, c(201, 27, 2))
  expect_equal_to_r(x[1, , ], cbind(rep(400, 27), rep(0, 27)))
  expect_equal_to_r(x[51, , ], cbind(rep(0, 27), rep(0, 27)))

  x <- torch::torch_stft(
    input = torch::torch_ones(3000),
    n_fft = 400,
    center = TRUE
  )

  expect_tensor_shape(x, c(201, 31, 2))
  expect_equal_to_r(x[1, , ], cbind(rep(400, 31), rep(0, 31)))
  expect_equal_to_r(x[51, , ], cbind(rep(0, 31), rep(0, 31)))

  x <- torch::torch_stft(
    input = torch::torch_ones(3000),
    n_fft = 400,
    center = TRUE,
    return_complex = TRUE
  )
  expect_equal(x$shape, c(201, 31))
  expect_true(x$dtype == torch_complex(real = 1, imag = 1)$dtype)

  x <- torch_stft(
    input = torch::torch_ones(3000),
    n_fft = 400,
    window = torch_ones(400),
    center = FALSE
  )

  expect_tensor_shape(x, c(201, 27, 2))
  expect_equal_to_r(x[1, , ], cbind(rep(400, 27), rep(0, 27)))
  expect_equal_to_r(x[51, , ], cbind(rep(0, 27), rep(0, 27)))
})

test_that("torch_one_hot", {
  expect_tensor_shape(torch_one_hot(torch_tensor(1L)), c(1, 1))
  expect_tensor_shape(torch_one_hot(torch_tensor(c(1L, 2L))), c(2, 2))
  expect_error(torch_one_hot(torch_tensor(0L)))
})

test_that("torch_split", {
  x <- torch_tensor(1:5)

  expect_length(torch_split(x, 2), 3)
  expect_length(torch_split(x, c(2, 3)), 2)

  expect_length(x$split(2), 3)
  expect_length(x$split(c(2, 3)), 2)
})

test_that("torch_nonzero", {
  x <- torch_tensor(c(0, 1, 2, 0, 3))
  expect_equal_to_r(torch_nonzero(x), matrix(c(2L, 3L, 5L), ncol = 1))
  expect_equal_to_r(x$nonzero(), matrix(c(2L, 3L, 5L), ncol = 1))

  o <- torch_nonzero(x, as_list = TRUE)
  expect_length(o, 1)
  expect_equal_to_r(o[[1]], c(2L, 3L, 5L))

  o <- x$nonzero(as_list = TRUE)
  expect_length(o, 1)
  expect_equal_to_r(o[[1]], c(2L, 3L, 5L))

  x <- torch_tensor(matrix(c(0, 1, 0, 1, 1, 0), nrow = 2))
  expect_equal(nrow(torch_nonzero(x)), 3)
  expect_equal(nrow(x$nonzero()), 3)

  o <- torch_nonzero(x, as_list = TRUE)
  expect_length(o, 2)

  o <- x$nonzero(as_list = TRUE)
  expect_length(o, 2)

  x <- torch_tensor(c(0, 0))
  expect_equal(nrow(torch_nonzero(x)), 0)
  expect_equal(nrow(x$nonzero()), 0)

  expect_equal(nrow(torch_nonzero(x, as_list = TRUE)[[1]]), 0)
  expect_equal(nrow(x$nonzero(as_list = TRUE)[[1]]), 0)

  skip_if_cuda_not_available()
  x <- torch_tensor(c(0, 1, 2, 0, 3), device = "cuda")
  expect_equal_to_r(torch_nonzero(x), matrix(c(2L, 3L, 5L), ncol = 1))
})

test_that("normal works", {
  x <- torch_normal(0, 1, size = c(2, 2))
  expect_tensor_shape(x, c(2, 2))
  expect_true(x$dtype == torch_float())

  x <- torch_normal(0, 1, size = c(2, 2), dtype = torch_float64())
  expect_true(x$dtype == torch_float64())

  x <- torch_normal(torch_zeros(2, 2), torch_ones(c(2, 2)))
  expect_tensor_shape(x, c(2, 2))

  x <- torch_normal(torch_zeros(2, 2), 1)
  expect_tensor_shape(x, c(2, 2))

  x <- torch_normal(1, torch_zeros(2, 2))
  expect_tensor_shape(x, c(2, 2))

  x <- torch_normal(mean = torch_zeros(2, 2))
  expect_tensor_shape(x, c(2, 2))

  x <- torch_normal(std = torch_zeros(2, 2))
  expect_tensor_shape(x, c(2, 2))

  x <- torch_normal(size = list(2, 2))
  expect_tensor_shape(x, c(2, 2))

  expect_error(torch_normal(torch_zeros(2), 1, c(2, 2)), class = "value_error")
  expect_error(torch_normal(1, torch_zeros(2), c(2, 2)), class = "value_error")
  expect_error(torch_normal(1, torch_zeros(2), dtype = torch_float64()), class = "value_error")
})

test_that("polygamma works", {
  a <- torch_tensor(c(1, 0.5))
  r <- torch_polygamma(1, a)
  expect_equal_to_r(a, c(1, 0.5))
  expect_equal_to_r(r, c(1.64493405818939, 4.93480205535889))
})

test_that("broadcast_shapes", {
  expect_equal(torch_broadcast_shapes(c(2, 2), c(2, 2)), c(2, 2))
  expect_equal(torch_broadcast_shapes(c(2, 1), c(2, 2)), c(2, 2))
  expect_error(torch_broadcast_shapes(c(2, 3), c(2, 2)))
})

test_that("tensordot works", {
  t1 <- torch_rand(10, 2, 3)
  t2 <- torch_rand(3, 3, 5)

  expect_tensor_shape(torch_tensordot(t1, t2, 1L), c(10, 2, 3, 5))
  expect_tensor_shape(torch_tensordot(t1, t2, list(3, 1)), c(10, 2, 3, 5))
})

test_that("multinomial works", {
  x <- torch_tensor(1)
  expect_equal_to_tensor(
    torch_multinomial(x, 10, replacement = TRUE),
    torch_ones(10)
  )
})
