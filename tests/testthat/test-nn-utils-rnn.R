to_int <- function(x) {
  x$to(dtype = torch_int())
}

test_that("pack_padded_sequence", {
  x <- torch_tensor(rbind(
    c(1, 2, 0, 0),
    c(1, 2, 3, 0),
    c(1, 2, 3, 4)
  ), dtype = torch_long())
  lens <- torch_tensor(c(2,3,4), dtype = torch_long())
  
  p <- nn_utils_rnn_pack_padded_sequence(x, lens, batch_first = TRUE, 
                                     enforce_sorted = FALSE)
  
  expect_equal_to_r(to_int(p$data), c(1, 1, 1, 2, 2, 2, 3, 3, 4))
  expect_equal_to_r(to_int(p$batch_sizes), c(3, 3, 2, 1))
  expect_equal_to_r(to_int(p$sorted_indices), c(2, 1, 0))
  expect_equal_to_r(to_int(p$unsorted_indices), c(2, 1, 0))
})

test_that("pack_sequence", {
  x <- torch_tensor(c(1,2,3), dtype = torch_long())
  y <- torch_tensor(c(4, 5), dtype = torch_long())
  z <- torch_tensor(c(6), dtype = torch_long())
  
  p <- nn_utils_rnn_pack_sequence(list(x, y, z))
  expect_equal_to_r(to_int(p$data), c(1, 4, 6, 2, 5, 3))
  expect_equal_to_r(to_int(p$batch_sizes), c(3, 2, 1))
})
