test_that("R7Class", {
  
  MyClass <- R7Class(
    "myclass",
    public = list(
      initialize = function(x) {
        self$x <- x
      },
      set_x = function(x) {
        self$x <- x
      },
      set_y2 = function(y) {
        private$set_y(y)
      }
    ),
    active = list(
      act = function() {
        self$x
      }
    ),
    private = list(
      set_y = function(y) {
        self$y <- y
      }
    )
  )
  
  s <- MyClass$new(x = 1)
  expect_equal(s$x, 1)
  
  s$set_x(2)
  expect_equal(s$x, 2)
  
  s$set_y2(4)
  expect_equal(s$y, 4)
  
  expect_equal(s$act, 2)
})
