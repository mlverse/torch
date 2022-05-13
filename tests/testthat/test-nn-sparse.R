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
  
  # a fixed seed should give the same weights as in python
  torch::torch_manual_seed(1)
  embedding <- nn_embedding_bag(10, 1)
  
  # this vector was obtained by running 
  # torch.manual_seed(1)
  # embedding = torch.nn.EmbeddingBag(10, 1)
  weights_py <-  c(0.6614,
                 0.2669,
                 0.0617,
                 0.6213,
                -0.4519,
                -0.1661,
                -1.5228,
                 0.3817,
                -1.0276,
                -0.5631)
  
  expect_equal_to_r(
    embedding$weight,
    matrix(weights_py, ncol = 1),
    tolerance = 1e-4
  )
  
  # input = torch.tensor([[1,2,4,5], [4,3,2,9]]) - 1
  # embedding(input)
  input <- torch_tensor(rbind(c(1, 2, 4, 5), c(4, 3, 2, 9)), dtype = torch_long())
  py_out <- c(0.2744, -0.0194)
  expect_equal_to_r(
    embedding(input),
    matrix(py_out, ncol = 1),
    tolerance = 1e-4
  )
  
})
