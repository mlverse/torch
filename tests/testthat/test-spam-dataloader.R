test_that("spam_dataloader loads and batches data correctly", {
  dl <- spam_dataloader(batch_size = 32, shuffle = TRUE)
  expect_true(inherits(dl, "dataloader"))
  
  iter <- dl$.iter()
  batch <- iter$.next()
  
  expect_equal(length(batch), 2)  # Should return a list with x and y
  expect_equal(batch[[1]]$dim()[2], 57)  # 57 features in x
  expect_true(batch[[2]]$dtype() == torch_long())  # Labels should be long dtype
})
