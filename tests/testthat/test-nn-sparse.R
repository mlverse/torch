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

test_that("nn_embedding_bag", {
  
  # an Embedding module containing 10 tensors of size 3
  embedding <- nn_embedding_bag(10, 3)
  # a batch of 2 samples of 4 indices each
  input <- torch_tensor(rbind(c(1, 2, 4, 5), c(4, 3, 2, 9)), dtype = torch_long())
  output <- embedding(input)
  
  expect_equal_to_tensor(output[1, ], (embedding$weight[1,] + embedding$weight[2,] + embedding$weight[4,] + embedding$weight[5,])/4)
  expect_equal_to_tensor(output[2, ], (embedding$weight[4,] + embedding$weight[3,] + embedding$weight[2,] + embedding$weight[9,])/4)
  
  # example with padding_idx
  embedding <- nn_embedding_bag(10, 3, padding_idx = 1, mode='sum')
  input <- torch_tensor(matrix(rbind(c(1, 3, 1, 6), c(2, 4, 1, 5)), nrow = 2), dtype = torch_long())
  output <- embedding(input)
  
  expect_equal_to_tensor(output[1, ], embedding$weight[3, ] + embedding$weight[6, ])
  expect_equal_to_tensor(output[2, ], embedding$weight[2, ] + embedding$weight[4, ] + embedding$weight[5, ])
})
