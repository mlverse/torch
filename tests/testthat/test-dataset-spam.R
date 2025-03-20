test_that("spam_dataset loads correctly", {
  skip_if_not_installed("torch")
  skip_on_cran()
  
  # Test dataset creation
  ds <- spam_dataset()
  
  # Check dataset length
  expect_equal(ds$.length(), 4601)
  
  # Check item structure
  item <- ds$.getitem(1)
  expect_length(item, 2)
  expect_true(is_torch_tensor(item[[1]]))
  expect_true(is_torch_tensor(item[[2]]))
  
  # Check data dimensions
  expect_equal(item[[1]]$shape, c(57))
  
  # Instead of checking exact shape, check it's a scalar tensor
  # (either empty shape or shape of length 1 with value 1)
  expect_true(length(item[[2]]$shape) == 0 || 
                (length(item[[2]]$shape) == 1 && item[[2]]$shape[1] == 1))
  
  expect_true(item[[2]]$dtype == torch_long())
})

test_that("spam_dataset transform works", {
  skip_if_not_installed("torch")
  skip_on_cran()
  
  # Test transform function
  normalize <- function(x) {
    (x - x$mean()) / x$std()
  }
  
  ds <- spam_dataset(transform = normalize)
  item <- ds$.getitem(1)
  
  # Normalized data should have mean close to 0 and std close to 1
  expect_true(abs(item[[1]]$mean()$item()) < 1e-5)
  expect_true(abs(item[[1]]$std()$item() - 1) < 1e-5)
})

test_that("spam_dataset works with dataloader", {
  skip_if_not_installed("torch")
  skip_on_cran()
  
  ds <- spam_dataset()
  dl <- dataloader(ds, batch_size = 32, shuffle = TRUE)
  
  # Check number of batches
  expect_equal(dl$.length(), ceiling(4601/32))
  
  # Check batch structure
  iter <- dl$.iter()
  batch <- iter$.next()
  
  expect_length(batch, 2)
  expect_equal(batch[[1]]$shape[2], 57)  # 57 features
  expect_true(batch[[1]]$shape[1] <= 32)  # Batch size
  expect_true(batch[[2]]$dtype == torch_long())  # Target is long tensor
})