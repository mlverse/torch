context("utils-data-dataloader")

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
  while(!is.null(batch <- dataloader_next(iter))) {
    expect_tensor(batch[[1]])
    expect_tensor(batch[[2]])
  }
  
  # iterating with an enum
  for (batch in enumerate(dl)) {
    expect_tensor(batch[[1]])
    expect_tensor(batch[[2]])    
  }
  
})

test_that("can have datasets that don't return tensors", {
  
  ds <- dataset(
    initialize = function() {},
    .getitem = function(index) {
      list(
        matrix(runif(10), ncol = 10),
        index,
        1:10
      )
    },
    .length = function() {
      100
    }
  )
  d <- ds()
  dl <- dataloader(d, batch_size = 32, drop_last = TRUE)
  
  # iterating with an enum
  for (batch in enumerate(dl)) {
    expect_tensor_shape(batch[[1]], c(32, 1, 10))
    expect_tensor_shape(batch[[2]], c(32))    
    expect_tensor_shape(batch[[3]], c(32, 10))    
  }
  
  expect_true(batch[[1]]$dtype == torch_float32())
  expect_true(batch[[2]]$dtype == torch_int64())
  expect_true(batch[[3]]$dtype == torch_int64())
  
})

test_that("dataloader that shuffles", {
  
  x <- torch_randn(100, 100)
  y <- torch_randn(100, 1)
  d <- tensor_dataset(x, y)
  dl <- dataloader(dataset = d, batch_size = 50, shuffle = TRUE)
  
  for(i in enumerate(dl))
    expect_tensor_shape(i[[1]], c(50, 100))

})

