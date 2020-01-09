test_that("Can create devices", {
  device <- torch_device("cuda")
  expect_equal(device$type, "cuda")
  expect_null(device$index)
  
  device <- torch_device("cuda:1")
  expect_equal(device$type, "cuda")
  expect_equal(device$index, 1)
  
  device <- torch_device("cuda", 1)
  expect_equal(device$type, "cuda")
  expect_equal(device$index, 1)
  
  device <- torch_device("cpu", 0)
  expect_equal(device$type, "cpu")
  expect_equal(device$index, 0)
})
