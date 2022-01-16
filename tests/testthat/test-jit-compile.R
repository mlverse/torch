test_that("jit compile works", {
  comp <- jit_compile("
  def fn (x):
    return torch.abs(x)

  def foo (x):
    return torch.sum(x)

  ")

  expect_equal_to_tensor(
    comp$fn(torch_tensor(-1)),
    torch_tensor(1)
  )
  expect_equal_to_tensor(
    comp$foo(torch_tensor(c(1, 2, 3))),
    torch_tensor(c(1, 2, 3))$sum()
  )
})

test_that("function are alive even if compilation unit dies", {
  comp <- jit_compile("
  def fn (x):
    return torch.abs(x)

  def foo (x):
    return torch.sum(x)

  ")

  fn <- comp$fn
  rm(comp)
  expect_error(
    {
      gc()
      gc() # this shouldn't trigger a delete on `comp` because `fn` is protecting it.
    },
    regexp = NA
  )

  expect_equal_to_tensor(
    fn(torch_tensor(-1)),
    torch_tensor(1)
  )

  rm(fn)
  expect_error(gc(), regexp = NA)
})
