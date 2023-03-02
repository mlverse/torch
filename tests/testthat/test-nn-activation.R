test_that("Sparsemax", {
  m <- nn_contrib_sparsemax(dim = -1)
  x <- rbind(
    c(0.01, 0.01, 0.01, 0.97),
    c(0.001, 0.01, 0.01, 0.97),
    c(0.01, 0.01, 0.01, 5)
  )

  x_t <- torch_tensor(x)

  pred <- m(x_t)

  expect_tensor_shape(m(x_t), c(3, 4))
  expect_equal_to_tensor(m(x_t)[1, ], x_t[1, ], tolerance = 1e-6)
  expect_equal_to_tensor(m(x_t)[2, ], torch_tensor(c(0.0033, 0.0123, 0.0123, 0.9723)), tolerance = 1e-4)
  expect_equal_to_tensor(m(x_t)[3, ], torch_tensor(c(0, 0, 0, 1)), tolerance = 1e-5)


  m <- nn_contrib_sparsemax(dim = -1)
  x <- rbind(
    c(0.01, 0.01, 0.01, 0.97),
    c(0.001, 0.01, 0.01, 0.97),
    c(0.01, 0.01, 0.01, 5)
  )

  x_t <- torch_tensor(x, requires_grad = TRUE)
  y <- torch_tensor(c(4L, 4L, 4L))
  l <- nnf_nll_loss(m(x_t), y)
  l$backward()

  expect_equal_to_tensor(
    x_t$grad,
    torch_tensor(
      rbind(
        c(0.0833, 0.0833, 0.0833, -0.2500),
        c(0.0833, 0.0833, 0.0833, -0.2500),
        c(0.0000, 0.0000, 0.0000, 0.0000)
      )
    ),
    tolerance = 1e-4
  )
})

test_that("Multihead attention works", {
  
  attn1 <- nn_multihead_attention(embed_dim = 10, num_heads = 1)
  attn2 <- nn_multihead_attention(embed_dim = 10, num_heads = 1, batch_first = TRUE)
  attn2$load_state_dict(attn1$state_dict())
  
  q <- torch_randn(5, 32, 10)
  k <- torch_randn(5, 32, 10)
  v <- torch_randn(5, 32, 10)
  
  res1 <- attn1(q, k, v)
  res2 <- attn2(q$transpose(2,1), k$transpose(2,1), v$transpose(2,1))
  
  expect_equal_to_tensor(res1[[1]], res2[[1]]$transpose(2,1))
  expect_equal_to_tensor(res1[[2]], res2[[2]])
  
  # comparing to python results.
  torch::torch_manual_seed(1)
  attn1 <- nn_multihead_attention(embed_dim = 2, num_heads = 1)
  x <- torch_randn(1,1,2)
  out <- attn1(x, x, x)
  
  expect_equal_to_r(out[[1]][1,1,], c(0.0736, -0.0599), tol = 1e-4)
  expect_equal_to_r(out[[2]][1,1,], c(1), tol = 1e-4)
  expect_equal_to_r(attn1$in_proj_weight[1,], c(-0.1782,  0.4406), tol = 1e-4)
  expect_equal_to_r(attn1$out_proj$weight[1,], c(0.3643, -0.3121), tol = 1e-4)
  
  # raise error when embed_dim is not divisible by num_heads.
  expect_error(nn_multihead_attention(embed_dim = 512, num_heads = 10), regexp="divisible")
})

test_that("silu works", {
  
  silu <- nn_silu()
  input <- torch_tensor(c(-1.0, 0.0, 1.0))
  expected_output <- torch_tensor(c(-0.26894142, 0.0, 0.73105858))

  expect_equal_to_tensor(silu(input), expected_output)
  
  silu <- nn_silu(inplace = TRUE)
  out <- silu(input)
  
  expect_equal_to_tensor(input, expected_output)
  
})
