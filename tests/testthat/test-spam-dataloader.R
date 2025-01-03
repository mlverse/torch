library(testthat)
library(torch)

test_that("spam_dataloader loads and batches data correctly", {
  # Create the dataloader with batch size 32 and shuffling enabled
  dl <- spam_dataloader(batch_size = 32, shuffle = TRUE, download = TRUE)
  
  # Check if the returned object is a dataloader
  expect_true(inherits(dl, "dataloader"), "The returned object is not a dataloader.")
  
  # Get the first batch
  iter <- dl$.iter()
  batch <- iter$.next()
  
  # Verify the batch structure
  expect_equal(length(batch), 2, "The batch should be a list with two elements (x and y).")
  expect_equal(batch[[1]]$dim()[2], 57, "The predictors (x) should have 57 features.")
  
  # Verify the data types
  expect_true(batch[[1]]$dtype() == torch_float(), "The predictors (x) should have dtype torch_float.")
  expect_true(batch[[2]]$dtype() == torch_long(), "The labels (y) should have dtype torch_long.")
  
  # Verify batch size
  expect_equal(batch[[1]]$size(1), 32, "The batch size for predictors (x) should match 32.")
  expect_equal(batch[[2]]$size(1), 32, "The batch size for labels (y) should match 32.")
})
