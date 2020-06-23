test_that("dataloader works", {
  x <- torch_randn(1000, 100)
  y <- torch_randn(1000, 1)
  dataset <- tensor_dataset(x, y)
  
  dl <- dataloader(dataset = dataset, batch_size = 32)
  expect_length(dl, 1000 %/% 32 + 1)
  
  expect_true(is_dataloader(dl))
  
  iter <- dl$.iter()
  b <- iter$.next()
  
  expect_tensor_shape(b[[1]], c(32, 100))
  expect_tensor_shape(b[[2]], c(32, 1))
  
  iter <- dl$.iter()
  for(i in 1:32)
    k <- iter$.next()
  
  expect_error(iter$.next(), class = "stop_iteration_error")
})

test_that("dataloader iteration", {
  
  x <- torch_randn(100, 100)
  y <- torch_randn(100, 1)
  dataset <- tensor_dataset(x, y)
  dl <- dataloader(dataset = dataset, batch_size = 32)
  
  # iterating with a while loop
  iter <- dataloader_make_iter(dl)
  while(!is.null(batch <- dataloader_get_next(iter))) {
    expect_tensor(batch[[1]])
    expect_tensor(batch[[2]])
  }
  
  # iterating with an enum
  for (batch in dataloader_enum(dl)) {
    expect_tensor(batch[[1]])
    expect_tensor(batch[[2]])    
  }
  
})

