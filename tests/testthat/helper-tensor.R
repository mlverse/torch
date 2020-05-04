#torch_zeros(1, names="hello") # trigger warning about named tensors

skip_if_not_test_examples <- function() {
  if (Sys.getenv("TEST_EXAMPLES", unset = "0") != "1")
    skip("Not testing examples/readme. Set the env var TEST_EXAMPLES = 1.")
}

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

expect_undefined_tensor <- function(object) {
  # TODO
}