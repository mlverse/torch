#torch_zeros(1, names="hello") # trigger warning about named tensors

expect_equal_to_tensor <- function(object, expected) {
  expect_equal(as_array(object), as_array(expected))
}

expect_not_equal_to_tensor <- function(object, expected) {
  expect_false(isTRUE(all.equal(as_array(object), as_array(expected))))
}

expect_no_error <- function(object, ...) {
  expect_error(object, NA, ...)
}

expect_tensor <- function(object) {
  expect_true(is_torch_tensor(object))
  expect_no_error(as_array(object))
}

expect_equal_to_r <- function(object, expected) {
  expect_equal(as_array(object), expected)
} 