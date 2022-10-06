skip_if_not_m1_mac()

test_that("can create tensors on the MPS device", {
  x <- torch_randn(10, 10, device = "mps")
  expect_tensor(x)
  expect_true(x$device == torch_device("mps", 0))
  
  y <- torch_mm(x, x)
  expect_true(y$device == torch_device("mps", 0))
})

test_that("can allocate a bunch of tensors without OOM", {
  expect_no_error({
    for(i in 1:25) x <- torch_randn(10000, 10000, device="mps")  
  })
})