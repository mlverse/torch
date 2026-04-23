test_that("all dataloader tests pass with SHM transport disabled", {
  withr::local_options(torch.dataloader_use_shm = FALSE)
  source(test_path("test-utils-data-dataloader.R"), local = TRUE)
})

test_that("all dataloader tests pass with SHM transport enabled", {
  skip_on_os("windows")
  withr::local_options(torch.dataloader_use_shm = TRUE)
  source(test_path("test-utils-data-dataloader.R"), local = TRUE)
})
