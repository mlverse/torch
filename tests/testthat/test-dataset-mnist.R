context("dataset-mnist")

test_that("tests for the mnist dataset", {
  
  # skipped on mac because I use mac and I don't want too download mnist evrytime
  skip_on_os("mac") 
  
  dir <- tempfile(fileext = "/")
  
  expect_error(
    ds <- mnist_dataset(dir)
  )
  
  ds <- mnist_dataset(dir, download = TRUE)
  i <- ds[1]
  
  expect_tensor(i[[1]])
  expect_tensor(i[[2]]$to(dtype = torch_int()))
  
  expect_tensor_shape(ds$data, c(60000, 28, 28))
  expect_tensor_shape(ds$targets$to(dtype = torch_int()), c(60000))
})


test_that("tests for the kmnist dataset", {
  
  # skipped on mac because I use mac and I don't want too download mnist evrytime
  skip_on_os("mac") 
  
  dir <- tempfile(fileext = "/")
  
  expect_error(
    ds <- kmnist_dataset(dir)
  )
  
  ds <- kmnist_dataset(dir, download = TRUE)
  i <- ds[1]
  
  expect_tensor(i[[1]])
  expect_tensor(i[[2]]$to(dtype = torch_int()))
  
  expect_tensor_shape(ds$data, c(60000, 28, 28))
  expect_tensor_shape(ds$targets$to(dtype = torch_int()), c(60000))
})