test_that("allocator calls gc", {
  gc()

  env <- new.env()
  trace(gc, print = FALSE, function() {
    env$gc_called <- TRUE
  })
  on.exit(untrace(gc))

  expect_error(torch_randn(1e9, 1e9))
  expect_true(env$gc_called)
})

test_that("gpu allocator is called", {
  skip_if_cuda_not_available()

  gc()

  env <- new.env()
  trace(quote(gc), print = FALSE, function() {
    env$gc_called <- TRUE
  })
  on.exit(untrace(gc))

  expect_error({
    torch_randn(1e6, 1e7, device = "cuda")
  })
  expect_true(env$gc_called)
})
