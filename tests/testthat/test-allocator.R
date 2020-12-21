test_that("allocator calls gc", {
  env <- new.env()
  trace(gc, print = FALSE, function() {
    env$gc_called <- TRUE
  })
  on.exit(untrace(gc))
  
  expect_error(torch_randn(1e9, 1e9))
  expect_true(env$gc_called)
})
