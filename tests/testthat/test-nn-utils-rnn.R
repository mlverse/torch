context("nn-utils-rnn")

to_int <- function(x) {
  x$to(dtype = torch_int())
}

test_that("pack_padded_sequence", {
  x <- torch_tensor(rbind(
    c(1, 2, 0, 0),
    c(1, 2, 3, 0),
    c(1, 2, 3, 4)
  ), dtype = torch_long())
  lens <- torch_tensor(c(2, 3, 4), dtype = torch_long())

  p <- nn_utils_rnn_pack_padded_sequence(x, lens,
    batch_first = TRUE,
    enforce_sorted = FALSE
  )

  expect_equal_to_r(to_int(p$data), c(1, 1, 1, 2, 2, 2, 3, 3, 4))
  expect_equal_to_r(to_int(p$batch_sizes), c(3, 3, 2, 1))
  expect_equal_to_r(to_int(p$sorted_indices), c(3, 2, 1))
  expect_equal_to_r(to_int(p$unsorted_indices), c(3, 2, 1))

  expect_error(nn_utils_rnn_pack_padded_sequence(x, lens,
    batch_first = TRUE,
    enforce_sorted = TRUE
  ))

  x <- torch_tensor(rbind(
    c(1, 2, 3, 4),
    c(1, 2, 3, 0),
    c(1, 2, 0, 0)
  ), dtype = torch_long())
  lens <- torch_tensor(c(4, 3, 2), dtype = torch_long())
  p <- nn_utils_rnn_pack_padded_sequence(x, lens,
    batch_first = TRUE,
    enforce_sorted = TRUE
  )

  expect_equal_to_r(to_int(p$data), c(1, 1, 1, 2, 2, 2, 3, 3, 4))
  expect_equal_to_r(to_int(p$batch_sizes), c(3, 3, 2, 1))
})

test_that("pack_sequence", {
  x <- torch_tensor(c(1, 2, 3), dtype = torch_long())
  y <- torch_tensor(c(4, 5), dtype = torch_long())
  z <- torch_tensor(c(6), dtype = torch_long())

  p <- nn_utils_rnn_pack_sequence(list(x, y, z))
  expect_equal_to_r(to_int(p$data), c(1, 4, 6, 2, 5, 3))
  expect_equal_to_r(to_int(p$batch_sizes), c(3, 2, 1))
})

test_that("pad_packed_sequence", {
  seq <- torch_tensor(rbind(
    c(1, 2, 0),
    c(3, 0, 0),
    c(4, 5, 6)
  ), dtype = torch_long())
  lens <- as.integer(c(2, 1, 3))
  packed <- nn_utils_rnn_pack_padded_sequence(seq, lens,
    batch_first = TRUE,
    enforce_sorted = FALSE
  )
  o <- nn_utils_rnn_pad_packed_sequence(packed, batch_first = TRUE)
  expect_equal_to_tensor(to_int(o[[1]]), to_int(seq))
  expect_equal_to_r(to_int(o[[2]]), lens)
})

test_that("pad_sequence", {
  x <- torch_ones(25, 300)
  y <- torch_ones(22, 300)
  z <- torch_ones(15, 300)

  o <- nn_utils_rnn_pad_sequence(list(x, y, z))
  expect_tensor_shape(o, c(25, 3, 300))
})
