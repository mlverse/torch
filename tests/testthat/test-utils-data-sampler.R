context("utils-data-sampler")

test_that("sampler's lenght", {
  x <- torch_randn(1000, 10)
  y <- torch_randn(1000)
  data <- tensor_dataset(x, y)

  sampler <- SequentialSampler$new(data)
  expect_length(sampler, 1000)

  sampler <- RandomSampler$new(data, num_samples = 10)
  expect_length(sampler, 10)

  sampler <- RandomSampler$new(data)
  expect_length(sampler, 1000)

  batch <- BatchSampler$new(sampler = sampler, batch_size = 32, drop_last = TRUE)
  expect_length(batch, 1000 %/% 32)

  batch <- BatchSampler$new(sampler = sampler, batch_size = 32, drop_last = FALSE)
  expect_length(batch, 1000 %/% 32 + 1)

  batch <- BatchSampler$new(sampler = sampler, batch_size = 100, drop_last = FALSE)
  expect_length(batch, 10)

  batch <- BatchSampler$new(sampler = sampler, batch_size = 1000, drop_last = FALSE)
  expect_length(batch, 1)

  batch <- BatchSampler$new(sampler = sampler, batch_size = 1001, drop_last = FALSE)
  expect_length(batch, 1)

  batch <- BatchSampler$new(sampler = sampler, batch_size = 1001, drop_last = TRUE)
  expect_length(batch, 0)
})

test_that("Random sampler, replacement = TRUE", {
  x <- torch_randn(2, 10)
  y <- torch_randn(2)
  data <- tensor_dataset(x, y)

  x <- RandomSampler$new(data, replacement = TRUE)
  it <- x$.iter()

  for (i in 1:length(x)) {
    k <- it()
    expect_true(k <= 2 && k >= 1)
  }

  expect_equal(it(), coro::exhausted())
})

test_that("Batch sampler", {
  x <- torch_randn(100, 10)
  y <- torch_randn(2)
  data <- tensor_dataset(x, y)

  r <- RandomSampler$new(data, replacement = FALSE)
  x <- BatchSampler$new(r, 32, TRUE)
  it <- x$.iter()

  expect_length(it(), 32)
  expect_length(it(), 32)
  expect_length(it(), 32)
  expect_equal(it(), coro::exhausted())
})
