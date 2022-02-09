test_that("max with indices", {
  x <- torch_tensor(c(5, 6, 7, 8))
  m <- torch_max(x, dim = 1)

  expect_equal_to_r(m[[2]]$to(dtype = torch_int()), 4)

  expect_equal_to_r(
    torch_max(c(2, 1), other = c(1, 2)),
    c(2, 2)
  )
})

test_that("min with indices", {
  x <- torch_tensor(c(5, 6, 7, 8))
  m <- torch_min(x, dim = 1)

  expect_equal_to_r(m[[2]]$to(dtype = torch_int()), 1)

  expect_equal_to_r(
    torch_min(c(2, 1), other = c(1, 2)),
    c(1, 1)
  )
})

test_that("argsort", {
  x <- torch_tensor(c(3, 2, 1))
  expect_equal_to_r(torch_argsort(x), c(3, 2, 1))
  expect_equal_to_r(x$argsort(), c(3, 2, 1))

  x <- torch_tensor(c(1, 2, 3))
  expect_equal_to_r(torch_argsort(x, descending = TRUE), c(3, 2, 1))
  expect_equal_to_r(x$argsort(descending = TRUE), c(3, 2, 1))

  x <- torch_tensor(1:10)$view(c(5, 2))
  expect_equal_to_r(torch_argsort(x, dim = 1)[, 1], 1:5)
  expect_equal_to_r(x$argsort(dim = 1)[, 1], 1:5)

  expect_equal_to_r(torch_argsort(x, dim = 2)[, 1], rep(1, 5))
  expect_equal_to_r(x$argsort(dim = 2)[, 1], rep(1, 5))
})

test_that("argmax", {
  x <- torch_tensor(c(1, 2, 3))
  expect_equal_to_r(torch_argmax(x), 3)
  expect_equal_to_r(x$argmax(), 3)

  x <- torch_tensor(c(3, 2, 1))
  expect_equal_to_r(torch_argmax(x), 1)
  expect_equal_to_r(x$argmax(), 1)

  x <- torch_tensor(1:9)$reshape(c(3, 3))
  expect_equal_to_r(torch_argmax(x, dim = 2), c(3, 3, 3))
  expect_equal(torch_argmax(x, dim = 2, keepdim = TRUE)$shape, c(3, 1))
})

test_that("argmin", {
  x <- torch_tensor(c(1, 2, 3))
  expect_equal_to_r(torch_argmin(x), 1)
  expect_equal_to_r(x$argmin(), 1)

  x <- torch_tensor(c(3, 2, 1))
  expect_equal_to_r(torch_argmin(x), 3)
  expect_equal_to_r(x$argmin(), 3)

  x <- torch_tensor(1:9)$reshape(c(3, 3))
  expect_equal_to_r(torch_argmin(x, dim = 2), c(1, 1, 1))
  expect_equal(torch_argmin(x, dim = 2, keepdim = TRUE)$shape, c(3, 1))
})

test_that("sort", {
  x <- torch_tensor(sample(1e2))
  expect_equal_to_r(torch_sort(x)[[2]], order(as.integer(x)))
  expect_equal_to_r(torch_sort(x, descending = TRUE)[[2]], order(as.integer(x), decreasing = TRUE))

  expect_equal_to_r(x$sort()[[2]], order(as.integer(x)))
  expect_equal_to_r(x$sort(descending = TRUE)[[2]], order(as.integer(x), decreasing = TRUE))
})
