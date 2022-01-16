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
