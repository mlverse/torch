test_that("can run convert to and back", {
  x <- list(torch_tensor(1))
  expect_equal(test_stack(x), x)

  x <- list(torch_tensor(1), 1)
  expect_equal(test_stack(x), x)

  x <- list(torch_tensor(1), 1, c(1, 2, 3))
  expect_equal(test_stack(x), x)

  x <- list(torch_tensor(1), 1, c(1, 2, 3), 1L, 1:10)
  expect_equal(test_stack(x), x)

  x <- list(torch_tensor(1), 1, c(1, 2, 3), 1L, 1:10, list(torch_tensor(1), torch_tensor(2)))
  expect_equal(test_stack(x), x)

  x <- list(torch_tensor(1), 1, c(1, 2, 3), list(1L, 1:10), list(torch_tensor(1), torch_tensor(2)))
  expect_equal(test_stack(x), x)

  x <- list(a = torch_tensor(1), b = torch_tensor(2))
  class(x) <- "jit_tuple"
  x <- list(x)
  expect_equal(test_stack(x), list(list(a = torch_tensor(1), b = torch_tensor(2))))
})
