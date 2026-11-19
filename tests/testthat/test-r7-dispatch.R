test_that("R7 active bindings and bound methods use the right receiver", {
  generator <- R7Class(
    "test_dispatch_receiver",
    public = list(
      initialize = function(value) {
        env <- new.env(parent = emptyenv())
        env$value <- value
        env
      },
      add = function(x) self$value + x,
      add_private = function(x) private$add(x)
    ),
    private = list(add = function(x) self$value + x),
    active = list(value = function(value) {
      if (missing(value)) get("value", envir = self) else assign("value", value, envir = self)
    })
  )
  on.exit(rm("test_dispatch_receiver", envir = .generators))
  x <- generator$new(2)
  y <- generator$new(10)
  add_x <- x$add
  expect_equal(x$value, 2)
  expect_equal(y[["value"]], 10)
  expect_equal(add_x(3), 5)
  expect_equal(y$add_private(3), 13)
  x$value <- 7
  expect_equal(x$value, 7)
  expect_equal(add_x(3), 10)
  expect_equal(y$value, 10)
  y[["value"]] <- 11
  expect_equal(y$value, 11)

  generator$set("public", "add", function(x) self$value - x)
  expect_equal(x$add(3), 4)
  expect_equal(add_x(3), 10)
})
