test_that("all dataloader tests pass with SHM transport disabled", {
  withr::local_options(torch.dataloader_use_shm = FALSE)
  source(test_path("test-utils-data-dataloader.R"), local = TRUE)
})

test_that("all dataloader tests pass with SHM transport enabled", {
  skip_on_os("windows")
  withr::local_options(torch.dataloader_use_shm = TRUE)
  source(test_path("test-utils-data-dataloader.R"), local = TRUE)
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
  result <- list(
    structure(list(name = shm1$name, nbytes = shm1$nbytes, shape = 10L, dtype = "float"),
              class = "torch_shared_tensor"),
    structure(list(name = shm2$name, nbytes = shm2$nbytes, shape = 10L, dtype = "float"),
              class = "torch_shared_tensor")
  )

  torch:::shm_unlink_recursive(result)

  expect_false(cpp_shm_exists(shm1$name))
  expect_false(cpp_shm_exists(shm2$name))
})

test_that("custom collate returning non-tensor objects works with SHM", {
  skip_on_os("windows")
  withr::local_options(torch.dataloader_use_shm = TRUE)

  ds <- dataset(
    initialize = function() {},
    .getitem = function(i) {
      list(x = torch_randn(3), label = paste0("item_", i))
    },
    .length = function() { 10 }
  )

  my_collate <- function(batch) {
    sapply(batch, function(b) b$label)
  }

  dl <- dataloader(ds(), batch_size = 5, num_workers = 1, collate_fn = my_collate)
  iter <- dataloader_make_iter(dl)
  batch <- dataloader_next(iter)

  expect_true(is.character(batch))
  expect_equal(length(batch), 5)
})

test_that("SHM preserves list class and names through roundtrip", {
  skip_on_os("windows")
  withr::local_options(torch.dataloader_use_shm = TRUE)

  ds <- dataset(
    initialize = function() {
      self$x <- matrix(rnorm(100), nrow = 10, ncol = 10)
    },
    .getitem = function(i) {
      list(x = torch_tensor(self$x[i, ]), y = i)
    },
    .length = function() { 10 }
  )

  my_collate <- function(batch) {
    out <- list(
      x = torch_stack(lapply(batch, function(b) b$x)),
      y = sapply(batch, function(b) b$y)
    )
    class(out) <- c("my_batch", "list")
    out
  }

  dl <- dataloader(ds(), batch_size = 5, num_workers = 1, collate_fn = my_collate)
  iter <- dataloader_make_iter(dl)
  batch <- dataloader_next(iter)

  expect_true(inherits(batch, "my_batch"))
  expect_named(batch, c("x", "y"))
  expect_tensor_shape(batch$x, c(5, 10))
})
