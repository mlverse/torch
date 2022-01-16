test_that("layer_norm", {
  input <- torch_tensor(t(matrix(1:3, ncol = 3, nrow = 3)), dtype = torch_float())

  m <- nn_layer_norm(3, elementwise_affine = TRUE)
  result <- matrix(
    c(
      -1.22473537921906, -1.22473537921906, -1.22473537921906,
      0, 0, 0, 1.22473549842834, 1.22473549842834, 1.22473549842834
    ),
    nrow = 3, ncol = 3
  )
  expect_equal_to_r(
    m(input),
    result,
    tolerance = 1e-6
  )

  m <- nn_layer_norm(3, elementwise_affine = FALSE)
  expect_equal_to_r(
    m(input),
    result,
    tolerance = 1e-6
  )

  input <- torch_randn(3, 4, 5)
  m <- nn_layer_norm(input$size()[-1])
  expect_tensor_shape(m(input), c(3, 4, 5))

  x <- torch_ones(5, 2)
  x[, 1] <- 0:4 * 10 * x[, 1]
  x[, 2] <- 1:5 * 10 * x[, 2]

  m <- nn_layer_norm(normalized_shape = 2)
  expect_equal_to_tensor(m(x), torch_cat(list(
    -torch_ones(5, 1),
    torch_ones(5, 1)
  ), dim = 2), tolerance = 1e-6)
})


test_that("group_norm", {
  input <- torch_tensor(t(matrix(1:3, ncol = 3, nrow = 3)), dtype = torch_float())

  m <- nn_layer_norm(3)
  mg <- nn_group_norm(1, 3)

  expect_equal_to_tensor(mg(input), m(input))
})
