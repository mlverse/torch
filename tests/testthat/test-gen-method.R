# test_that("__and__", {
#   x <- torch_tensor(TRUE)
#   y <- x$`__and__`(x)
#   expect_tensor(y)
#   expect_equal_to_tensor(y, x)
#   
#   x <- torch_tensor(c(TRUE, FALSE))
#   y <- x$`__and__`(TRUE)
#   expect_tensor(y)
#   expect_equal_to_tensor(y, x)
# })
