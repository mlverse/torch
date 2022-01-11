context("indexing")

test_that("[ works", {
  x <- torch_randn(c(10, 10, 10))
  expect_equal(as_array(x[1, 1, 1]), as_array(x)[1, 1, 1])
  expect_equal(as_array(x[1, , ]), as_array(x)[1, , ])
  expect_equal(as_array(x[1:5, , ]), as_array(x)[1:5, , ])
  expect_equal(as_array(x[1:10:2, , ]), as_array(x)[seq(1, 10, by = 2), , ])

  x <- torch_tensor(0:9)
  expect_equal(as_array(x[-1]$to(dtype = torch_int())), 9)
  expect_equal(as_array(x[-2:10]$to(dtype = torch_int())), c(8, 9))
  expect_equal(as_array(x[2:N]$to(dtype = torch_int())), c(1:9))

  x <- torch_randn(c(10, 10, 10, 10))
  expect_equal(as_array(x[1, ..]), as_array(x)[1, , , ])
  expect_equal(as_array(x[1, 1, ..]), as_array(x)[1, 1, , ])
  expect_equal(as_array(x[.., 1]), as_array(x)[, , , 1])
  expect_equal(as_array(x[.., 1, 1]), as_array(x)[, , 1, 1])

  x <- torch_randn(c(10, 10, 10, 10))
  i <- c(1, 2, 3, 4)
  expect_equal(as_array(x[!!!i]), as_array(x)[1, 2, 3, 4])
  i <- c(1, 2)
  expect_equal(as_array(x[!!!i, 3, 4]), as_array(x)[1, 2, 3, 4])

  x <- torch_tensor(1:10)
  y <- 1:10
  expect_equal_to_r(x[c(1, 3, 2, 5)]$to(dtype = torch_int()), y[c(1, 3, 2, 5)])

  index <- 1:3
  expect_equal_to_r(x[index]$to(dtype = torch_int()), y[index])

  x <- torch_randn(10, 10)
  x[c(2, 3, 1), c(3, 2, 1)]
  expect_equal_to_r(x[c(2, 3, 1), c(3, 2, 1)], as_array(x)[c(2, 3, 1), c(3, 2, 1)])

  x <- torch_randn(10)
  expect_equal_to_tensor(x[1:5, ..], x[1:5])

  x <- torch_randn(10)
  expect_tensor_shape(x[, NULL], c(10, 1))
  expect_tensor_shape(x[NULL, , NULL], c(1, 10, 1))
  expect_tensor_shape(x[NULL, , NULL, NULL], c(1, 10, 1, 1))

  x <- torch_randn(10)
  expect_tensor_shape(x[, newaxis], c(10, 1))
  expect_tensor_shape(x[newaxis, , newaxis], c(1, 10, 1))
  expect_tensor_shape(x[newaxis, , newaxis, newaxis], c(1, 10, 1, 1))

  x <- torch_randn(10, 10)
  expect_tensor_shape(x[1, , drop = FALSE], c(1, 10))
  expect_tensor_shape(x[.., 1, drop = FALSE], c(10, 1))
  expect_tensor_shape(x[.., -1, drop = FALSE], c(10, 1))
})

test_that("indexing error expectations", {
  x <- torch_randn(c(10, 10, 10, 10))
  expect_error(x[1, 1, 1, 1, 1])
  x <- torch_tensor(10)
  expect_error(x[0])
  expect_error(x[c(0, 1)])
})

test_that("indexing with boolean tensor", {
  x <- torch_tensor(c(-1, -2, 0, 1, 2))
  expect_equal_to_r(x[x < 0], c(-1, -2))

  x <- torch_tensor(rbind(
    c(-1, -2, 0, 1, 2),
    c(2, 1, 0, -1, -2)
  ))

  expect_equal_to_r(x[x < 0], c(-1, -2, -1, -2))

  expect_error(x[x < 0, 1])
})

test_that("slice with negative indexes", {
  x <- torch_tensor(c(1, 2, 3))
  expect_equal_to_r(x[2:-1], c(2, 3))
  expect_equal_to_r(x[-2:-1], c(2, 3))
  expect_equal_to_r(x[-3:-2], c(1, 2))

  expect_equal_to_r(x[c(-1, -2)], c(3, 2))
})

test_that("subset assignment", {
  x <- torch_randn(2, 2)
  x[1, 1] <- torch_tensor(0)
  x
  expect_equal_to_r(x[1, 1], 0)

  x[1, 2] <- 0
  expect_equal_to_r(x[1, 2], 0)

  x[1, 2] <- 1L
  expect_equal_to_r(x[1, 2], 1)

  x <- torch_tensor(c(TRUE, FALSE))
  x[2] <- TRUE
  expect_equal_to_r(x[2], TRUE)

  x <- torch_tensor(rbind(
    c(-1, -2, 0, 1, 2),
    c(2, 1, 0, -1, -2)
  ))

  x[x <= 0] <- 1
  expect_true(as_array(torch_all(x > 0)))

  x <- torch_tensor(c(1, 2, 3, 4, 5))
  x[1:2] <- c(0, 0)
  expect_equal_to_r(x[1:2], c(0, 0))
})

test_that("indexing with R boolean vectors", {
  x <- torch_tensor(c(1, 2))
  expect_equal_to_r(x[TRUE], matrix(c(1, 2), nrow = 1))
  expect_equal_to_r(x[FALSE], matrix(data = 1, ncol = 2, nrow = 0))
  expect_equal_to_r(x[c(TRUE, FALSE)], 1)
})

test_that("indexing with long tensors", {
  x <- torch_randn(4, 4)
  index <- torch_tensor(1, dtype = torch_long())
  expect_equal(x[index, index]$item(), x[1, 1]$item())
  expect_tensor_shape(x[index, index], c(1, 1))

  index <- torch_scalar_tensor(1, dtype = torch_long())
  expect_equal_to_tensor(x[index, index], x[1, 1])

  index <- torch_tensor(-1, dtype = torch_long())
  expect_equal(x[index, index]$item(), x[-1, -1]$item())
  expect_tensor_shape(x[index, index], c(1, 1))

  index <- torch_scalar_tensor(-1, dtype = torch_long())
  expect_equal_to_tensor(x[index, index], x[-1, -1])

  index <- torch_tensor(c(-1, 1), dtype = torch_long())
  expect_equal_to_tensor(x[index, index], x[c(-1, 1), c(-1, 1)])

  index <- torch_tensor(c(-1, 0, 1), dtype = torch_long())
  expect_error(x[index, ], regexp = "Indexing starts at 1")
})

test_that("can use the slc construct", {
  x <- torch_randn(10, 10)
  r <- as_array(x)

  expect_equal_to_r(
    x[slc(start = 1, end = 5, step = 2), ],
    r[seq(1, 5, by = 2), ]
  )

  expect_equal_to_r(
    x[slc(start = 1, end = 5, step = 2), 1],
    r[seq(1, 5, by = 2), 1]
  )

  expect_equal_to_r(
    x[slc(start = 1, end = 5, step = 2), slc(start = 1, end = 5, step = 2)],
    r[seq(1, 5, by = 2), seq(1, 5, by = 2)]
  )

  expect_equal_to_tensor(
    x[slc(2, Inf), ],
    x[2:N, ]
  )
})

test_that("print slice", {
  testthat::local_edition(3)
  expect_snapshot(print(slc(1, 3, 5)))
})

test_that("mix vector indexing with slices and others", {
  x <- torch_randn(3, 3, 3)
  expect_equal_to_tensor(
    x[c(1, 2), 1:2, c(1, 2)],
    x[1:2, 1:2, 1:2]
  )

  expect_equal_to_tensor(
    x[c(1, 2), newaxis, 1:2, c(1, 2)],
    x[1:2, newaxis, 1:2, 1:2]
  )

  expect_equal_to_tensor(
    x[newaxis, c(1, 2), newaxis, 1:2, c(1, 2)],
    x[newaxis, 1:2, newaxis, 1:2, 1:2]
  )

  expect_equal_to_tensor(
    x[c(1, 2), c(1, 2), ],
    x[1:2, 1:2, ]
  )

  expect_equal_to_tensor(
    x[c(1, 2), , c(1, 2)],
    x[1:2, , 1:2]
  )

  expect_equal_to_tensor(
    x[c(1, 2), c(1, 2), c(1, 2)],
    x[1:2, 1:2, 1:2]
  )

  expect_equal_to_tensor(
    x[c(1, 2), c(1, 2), newaxis, c(1, 2)],
    x[1:2, 1:2, newaxis, 1:2]
  )
})

test_that("boolean vector indexing works as expected", {
  x <- torch_randn(4, 4, 4)
  index <- c(TRUE, FALSE, TRUE, FALSE)

  expect_equal_to_r(
    x[index, index, index],
    as_array(x)[index, index, index]
  )
})

test_that("regression test for #691", {
  a <- torch_randn(c(6, 4))
  b <- c(1, 2, 3)
  a[b]
  expect_equal(b, c(1, 2, 3))
})

test_that("regression test for #695", {
  a <- torch_randn(c(3, 4, 2))
  b <- torch_tensor(c(1, 3), dtype = torch_long())

  expect_equal_to_r(
    a[.., b, ],
    as.array(a)[, c(1, 3), ]
  )

  a <- torch_randn(c(3, 4, 3))

  expect_equal_to_r(
    a[.., b, b],
    as.array(a)[, c(1, 3), c(1, 3)]
  )

  expect_equal_to_r(
    a[b, .., b],
    as.array(a)[c(1, 3), , c(1, 3)]
  )
})
