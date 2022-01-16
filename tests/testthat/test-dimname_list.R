context("dimname-list")

test_that("dimname can be created", {
  x <- torch_dimname("hello")
  expect_output(print(x))
})

test_that("dimname lists can be created", {
  x <- torch_dimname_list(c("hello", "world"))
  expect_output(print(x))

  x <- torch_dimname_list(letters)
  expect_output(print(x))
})
