test_that("dataloader works", {
  x <- torch_randn(1000, 100)
  y <- torch_randn(1000, 1)
  dataset <- utils_dataset_tensor(x, y)
  
  dl <- utils_dataloader(dataset = dataset, batch_size = 32)
  expect_length(dl, 1000 %/% 32 + 1)
  
  iter <- dl$.iter()
  b <- iter$.next()
  
  expect_tensor_shape(b[[1]], c(32, 100))
  expect_tensor_shape(b[[2]], c(32, 1))
  
  iter <- dl$.iter()
  for(i in 1:32)
    k <- iter$.next()
  
  expect_error(iter$.next(), class = "stop_iteration_error")
})
