test_that("save tensor", {
  fname <- tempfile(fileext = "pt")
  x <- torch_randn(10, 10)
  torch_save(x, fname)
  y <- torch_load(fname)
  
  expect_equal_to_tensor(x, y)
})
