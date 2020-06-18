to_int <- function(x) {
  x$to(dtype = torch_int())
}

test_that("pack padded sequences", {
  x <- torch_tensor(rbind(
    c(1, 2, 0, 0),
    c(1, 2, 3, 0),
    c(1, 2, 3, 4)
  ), dtype = torch_long())
  lens <- torch_tensor(c(2,3,4), dtype = torch_long())
  
  p <- nn_utils_rnn_pack_padded_sequences(x, lens, batch_first = TRUE, 
                                     enforce_sorted = FALSE)
  
  expect_equal_to_r(to_int(p$data), c(1, 1, 1, 2, 2, 2, 3, 3, 4))
  expect_equal_to_r(to_int(p$batch_sizes), c(3, 3, 2, 1))
  expect_equal_to_r(to_int(p$sorted_indices), c(2, 1, 0))
  expect_equal_to_r(to_int(p$unsorted_indices), c(2, 1, 0))
})
