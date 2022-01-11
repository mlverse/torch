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
  for (i in 1:32) {
    k <- iter$.next()
  }

  expect_equal(iter$.next(), coro::exhausted())
})

test_that("dataloader iteration", {
  x <- torch_randn(100, 100)
  y <- torch_randn(100, 1)
  dataset <- tensor_dataset(x, y)
  dl <- dataloader(dataset = dataset, batch_size = 32)

  # iterating with a while loop
  iter <- dataloader_make_iter(dl)
  while (!is.null(batch <- dataloader_next(iter))) {
    expect_tensor(batch[[1]])
    expect_tensor(batch[[2]])
  }

  expect_warning(class = "deprecated", {
    # iterating with an enum
    for (batch in enumerate(dl)) {
      expect_tensor(batch[[1]])
      expect_tensor(batch[[2]])
    }
  })
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
  expect_warning(class = "deprecated", {
    for (batch in enumerate(dl)) {
      expect_tensor_shape(batch[[1]], c(32, 1, 10))
      expect_true(batch[[1]]$dtype == torch_float())
      expect_tensor_shape(batch[[2]], c(32))

      expect_tensor_shape(batch[[3]], c(32, 10))
      expect_true(batch[[3]]$dtype == torch_long())
    }
  })
  expect_true(batch[[1]]$dtype == torch_float32())
  expect_true(batch[[2]]$dtype == torch_int64())
  expect_true(batch[[3]]$dtype == torch_int64())
})

test_that("dataloader that shuffles", {
  x <- torch_randn(100, 100)
  y <- torch_randn(100, 1)
  d <- tensor_dataset(x, y)
  dl <- dataloader(dataset = d, batch_size = 50, shuffle = TRUE)

  expect_warning(class = "deprecated", {
    for (i in enumerate(dl)) {
      expect_tensor_shape(i[[1]], c(50, 100))
    }
  })

  dl <- dataloader(dataset = d, batch_size = 30, shuffle = TRUE)
  j <- 0
  expect_warning(class = "deprecated", {
    for (i in enumerate(dl)) {
      j <- j + 1
      if (j == 4) {
        expect_tensor_shape(i[[1]], c(10, 100))
      } else {
        expect_tensor_shape(i[[1]], c(30, 100))
      }
    }
  })
})


test_that("named outputs", {
  ds <- dataset(
    initialize = function() {

    },
    .getitem = function(i) {
      list(x = i, y = 2 * i)
    },
    .length = function() {
      1000
    }
  )()

  expect_named(ds[1], c("x", "y"))

  dl <- dataloader(ds, batch_size = 4)
  iter <- dataloader_make_iter(dl)

  expect_named(dataloader_next(iter), c("x", "y"))
})

test_that("can use a dataloader with coro", {
  ds <- dataset(
    initialize = function() {

    },
    .getitem = function(i) {
      list(x = i, y = 2 * i)
    },
    .length = function() {
      10
    }
  )()

  expect_named(ds[1], c("x", "y"))

  dl <- dataloader(ds, batch_size = 5)
  j <- 1
  loop(for (batch in dl) {
    expect_named(batch, c("x", "y"))
    expect_tensor_shape(batch$x, 5)
    expect_tensor_shape(batch$y, 5)
  })
})

test_that("dataloader works with num_workers", {
  if (cuda_is_available()) {
    skip_on_os("windows")
  }

  ds <- dataset(
    .length = function() {
      20
    },
    initialize = function() {},
    .getitem = function(id) {
      list(x = .worker_info$id)
    }
  )

  dl <- dataloader(ds(), batch_size = 10, num_workers = 2)

  i <- 1
  expect_warning(class = "deprecated", {
    for (batch in enumerate(dl)) {
      expect_equal_to_tensor(batch$x, i * torch_ones(10))
      i <- i + 1
    }
  })
})

test_that("dataloader catches errors on workers", {
  if (cuda_is_available()) {
    skip_on_os("windows")
  }

  ds <- dataset(
    .length = function() {
      20
    },
    initialize = function() {},
    .getitem = function(id) {
      stop("the error id is 5567")
      list(x = .worker_info$id)
    }
  )

  dl <- dataloader(ds(), batch_size = 10, num_workers = 2)
  iter <- dataloader_make_iter(dl)

  expect_error(
    dataloader_next(iter),
    class = "runtime_error",
    regexp = "5567"
  )
})

test_that("woprker init function is respected", {
  if (cuda_is_available()) {
    skip_on_os("windows")
  }

  ds <- dataset(
    .length = function() {
      20
    },
    initialize = function() {},
    .getitem = function(id) {
      list(x = theid)
    }
  )

  worker_init_fn <- function(id) {
    theid <<- id * 2
  }

  dl <- dataloader(ds(),
    batch_size = 10, num_workers = 2,
    worker_init_fn = worker_init_fn
  )

  i <- 1
  expect_warning(class = "deprecated", {
    for (batch in enumerate(dl)) {
      expect_equal_to_tensor(batch$x, i * 2 * torch_ones(10))
      i <- i + 1
    }
  })
})

test_that("dataloader timeout is respected", {
  if (cuda_is_available()) {
    skip_on_os("windows")
  }

  ds <- dataset(
    .length = function() {
      20
    },
    initialize = function() {},
    .getitem = function(id) {
      Sys.sleep(10)
      list(x = 1)
    }
  )

  dl <- dataloader(ds(),
    batch_size = 10, num_workers = 2,
    timeout = 5
  ) # (timeout is in miliseconds)

  iter <- dataloader_make_iter(dl)
  expect_error(
    dataloader_next(iter),
    class = "runtime_error",
    regexp = "timed out"
  )
})

test_that("can return tensors in multiworker dataloaders", {
  if (cuda_is_available()) {
    skip_on_os("windows")
  }

  ds <- dataset(
    .length = function() {
      20
    },
    initialize = function() {},
    .getitem = function(id) {
      list(x = torch_scalar_tensor(1))
    }
  )

  dl <- dataloader(ds(), batch_size = 10, num_workers = 2)

  expect_warning(class = "deprecated", {
    for (batch in enumerate(dl)) {
      expect_equal_to_tensor(batch$x, torch_ones(10))
    }
  })
})

test_that("can make reproducible runs", {
  if (cuda_is_available()) {
    skip_on_os("windows")
  }

  ds <- dataset(
    .length = function() {
      20
    },
    initialize = function() {},
    .getitem = function(id) {
      list(x = runif(1), y = torch_randn(1))
    }
  )

  dl <- dataloader(ds(), batch_size = 10, num_workers = 2)

  set.seed(1)
  iter <- dataloader_make_iter(dl)
  b1 <- dataloader_next(iter)

  set.seed(1)
  iter <- dataloader_make_iter(dl)
  b2 <- dataloader_next(iter)

  expect_equal(b1$x, b2$x)
  expect_equal_to_tensor(b1$y, b2$y)
})

test_that("load packages in dataloader", {
  ds <- dataset(
    .length = function() {
      20
    },
    initialize = function() {},
    .getitem = function(id) {
      torch_tensor("coro" %in% (.packages()))
    }
  )

  dl <- dataloader(ds(), batch_size = 10, num_workers = 2)

  iter <- dataloader_make_iter(dl)
  b1 <- dataloader_next(iter)

  expect_equal(torch_any(b1)$item(), FALSE)

  dl <- dataloader(ds(), batch_size = 10, num_workers = 2, worker_packages = "coro")

  iter <- dataloader_make_iter(dl)
  b1 <- dataloader_next(iter)

  expect_equal(torch_all(b1)$item(), TRUE)
})

test_that("globals can be found", {
  ds <- dataset(
    .length = function() {
      20
    },
    initialize = function() {},
    .getitem = function(id) {
      hello_fn()
    }
  )

  dl <- dataloader(ds(), batch_size = 10, num_workers = 2)

  iter <- dataloader_make_iter(dl)
  expect_error(
    b1 <- dataloader_next(iter)
  )

  expect_error(
    dl <- dataloader(ds(),
      batch_size = 10, num_workers = 2,
      worker_globals = c("hello", "world")
    ),
    class = "runtime_error"
  )

  hello_fn <- function() {
    torch_randn(5, 5)
  }

  dl <- dataloader(ds(), batch_size = 10, num_workers = 2, worker_globals = list(
    hello_fn = hello_fn
  ))

  iter <- dataloader_make_iter(dl)
  expect_tensor_shape(dataloader_next(iter), c(10, 5, 5))

  dl <- dataloader(ds(),
    batch_size = 10, num_workers = 2,
    worker_globals = "hello_fn"
  )
  iter <- dataloader_make_iter(dl)
  expect_tensor_shape(dataloader_next(iter), c(10, 5, 5))
})

test_that("datasets can use an optional .getbatch method for speedups", {
  d <- dataset(
    initialize = function() {},
    .getbatch = function(indexes) {
      list(
        torch_randn(length(indexes), 10),
        torch_randn(length(indexes), 1)
      )
    },
    .length = function() {
      100
    }
  )

  dl <- dataloader(d(), batch_size = 10)
  coro::loop(for (x in dl) {
    expect_length(x, 2)
    expect_tensor_shape(x[[1]], c(10, 10))
    expect_tensor_shape(x[[2]], c(10, 1))
  })
})

test_that("dataloaders handle .getbatch that don't necessarily return a torch_tensor", {
  d <- dataset(
    initialize = function() {},
    .getbatch = function(indexes) {
      list(
        array(0, dim = c(length(indexes), 10)),
        array(0, dim = c(length(indexes), 1))
      )
    },
    .length = function() {
      100
    }
  )

  dl <- dataloader(d(), batch_size = 10)
  coro::loop(for (x in dl) {
    expect_length(x, 2)
    expect_tensor_shape(x[[1]], c(10, 10))
    expect_tensor_shape(x[[2]], c(10, 1))
  })
})

test_that("a value error is returned when its not possible to convert", {
  d <- dataset(
    initialize = function() {},
    .getbatch = function(indexes) {
      "a"
    },
    .length = function() {
      100
    }
  )

  expect_error(
    dataloader_next(dataloader_make_iter(dataloader(d(), batch_size = 10))),
    regexp = "Can't convert data of class.*",
    class = "value_error"
  )
})


test_that("warning tensor", {
  dt <- dataset(
    initialize = function() {
      self$x <- torch_randn(100, 100)
      private$k <- torch_randn(10, 10)
      self$z <- list(
        k = torch_tensor(1),
        torch_tensor(2)
      )
    },
    .getitem = function(i) {
      torch_randn(1, 1)
    },
    .length = function() {
      100
    },
    active = list(
      y = function() {
        torch_randn(1)
      }
    ),
    private = list(
      k = 1
    )
  )


  dt <- dt()
  expect_warning(
    x <- dataloader(dt, batch_size = 2, num_workers = 10),
    regexp = "parallel dataloader"
  )
})
