context("readme")

test_that("can render readme.Rmd", {
  skip_if_not_test_examples()
  x <- tempfile(fileext = "md")
  k <- knitr::knit("../../README.Rmd", output = x, quiet = TRUE, envir = new.env())
  expect_equal(x, k)
})
