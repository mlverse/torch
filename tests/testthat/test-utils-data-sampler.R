test_that("sampler's lenght", {
  x <- torch_randn(1000, 10)
  y <- torch_randn(1000)
  data <- tensor_dataset(x, y)
  
  sampler <- SequentialSampler$new(data)
  expect_length(sampler, 1000)
  
  sampler <- RandomSampler$new(data)
  expect_length(sampler, 1000)
  
  batch <- BatchSampler$new(sampler = sampler, batch_size = 32, drop_last = TRUE)
  expect_length(batch, 1000 %/% 32)
  
  batch <- BatchSampler$new(sampler = sampler, batch_size = 32, drop_last = FALSE)
  expect_length(batch, 1000 %/% 32 + 1)
})

