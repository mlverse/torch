test_that("all dataloader tests pass with SHM transport", {
  skip_on_os("windows")
  withr::local_options(torch.dataloader_use_shm = TRUE)
  source(test_path("test-utils-data-dataloader.R"), local = TRUE)
})
