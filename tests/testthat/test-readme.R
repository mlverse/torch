test_that("can render readme.Rmd", {
  x <- tempfile(fileext = "md")
  k <- knitr::knit("../../README.Rmd", output = x, quiet = TRUE, envir = new.env())
  expect_equal(x, k)
})
