test_that("nn_embedding", {

  # an Embedding module containing 10 tensors of size 3
  embedding <- nn_embedding(10, 3)
  # a batch of 2 samples of 4 indices each
  input <- torch_tensor(rbind(c(1, 2, 4, 5), c(4, 3, 2, 9)), dtype = torch_long())
  output <- embedding(input)

  expect_equal_to_tensor(output[1, 1, ], embedding$weight[1, ])

  # example with padding_idx
  embedding <- nn_embedding(10, 3, padding_idx = 1)
  input <- torch_tensor(matrix(c(1, 3, 1, 6), nrow = 1), dtype = torch_long())
  output <- embedding(input)

  expect_equal_to_tensor(output[1, 1, ], embedding$weight[1, ])
})
