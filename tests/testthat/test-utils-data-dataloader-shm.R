test_that("all dataloader tests pass with SHM transport disabled", {
  withr::local_options(torch.dataloader_use_shm = FALSE)
  source(test_path("test-utils-data-dataloader.R"), local = TRUE)
})

test_that("all dataloader tests pass with SHM transport enabled", {
  skip_on_os("windows")
  withr::local_options(torch.dataloader_use_shm = TRUE)
  source(test_path("test-utils-data-dataloader.R"), local = TRUE)
})

test_that("in-place ops work on SHM-backed tensors", {
  skip_on_os("windows")
  withr::local_options(torch.dataloader_use_shm = TRUE)

  ds <- dataset(
    initialize = function() {
      self$x <- matrix(1:20, nrow = 4, ncol = 5)
    },
    .getitem = function(i) {
      torch_tensor(self$x[i, ], dtype = torch_float())
    },
    .length = function() { 4 }
  )

  dl <- dataloader(ds(), batch_size = 2, num_workers = 1)
  iter <- dataloader_make_iter(dl)
  batch <- dataloader_next(iter)

  # in-place ops must not segfault on SHM-backed tensors
  expect_no_error(batch$add_(1))
  expect_no_error(batch$div_(2))
  expect_no_error(batch$mul_(3))
})

test_that("SHM segments from unconsumed batches are cleaned up", {
  skip_on_os("windows")

  # Create SHM segments as a worker would
  t <- torch_randn(10)
  shm1 <- cpp_tensor_to_shm(t)
  shm2 <- cpp_tensor_to_shm(t)

  expect_true(cpp_shm_exists(shm1$name))
  expect_true(cpp_shm_exists(shm2$name))

  # shm_unlink_recursive walks a shared result and unlinks all segments
  result <- structure(list(
    structure(list(name = shm1$name, nbytes = shm1$nbytes, shape = 10L, dtype = "float"),
              class = "torch_shared_tensor"),
    structure(list(name = shm2$name, nbytes = shm2$nbytes, shape = 10L, dtype = "float"),
              class = "torch_shared_tensor")
  ), class = c("torch_shared_batch", "list"))

  torch:::shm_unlink_recursive(result)

  expect_false(cpp_shm_exists(shm1$name))
  expect_false(cpp_shm_exists(shm2$name))
})
