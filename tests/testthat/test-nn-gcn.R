test_that("nn_gcn_layer can be initialized", {
  layer <- nn_gcn_layer(in_features = 10, out_features = 5)
  expect_s3_class(layer, "nn_module")
  expect_s3_class(layer, "nn_gcn_layer")
  expect_length(layer$parameters, 3)
})

test_that("nn_gcn_layer forward pass produces correct dimensions", {
  n_nodes <- 4
  in_features <- 3
  out_features <- 2
  layer <- nn_gcn_layer(in_features, out_features)
  x <- torch_randn(n_nodes, in_features)
  adj_norm <- torch_eye(n_nodes)
  output <- layer(x, adj_norm)
  expect_tensor_shape(output, c(n_nodes, out_features))
})

test_that("nn_gcn_layer parameters have gradients after backward", {
  layer <- nn_gcn_layer(in_features = 5, out_features = 3)
  x <- torch_randn(2, 5)
  adj <- torch_eye(2)
  output <- layer(x, adj)
  loss <- output$sum()
  loss$backward()
  expect_false(is.null(layer$theta$weight$grad))
  expect_false(is.null(layer$phi$weight$grad))
  expect_false(is.null(layer$psi$grad))
})

test_that("nn_gcn_layer aggregates neighbor features correctly", {
  layer <- nn_gcn_layer(in_features = 2, out_features = 2)
  with_no_grad({
    layer$theta$weight$copy_(torch_eye(2))
    layer$phi$weight$copy_(torch_eye(2))
    layer$psi$copy_(torch_zeros(1, 2))
  })
  x <- torch_tensor(matrix(c(1, 0, 0, 1), nrow = 2, byrow = TRUE))
  adj_norm <- torch_tensor(matrix(c(0, 1, 1, 0), nrow = 2, byrow = TRUE))
  output <- layer(x, adj_norm)
  expected <- torch_ones(2, 2)
  expect_equal_to_tensor(output, expected, tolerance = 1e-5)
})

test_that("nn_gcn_layer handles single node graph", {
  layer <- nn_gcn_layer(in_features = 4, out_features = 2)
  x <- torch_randn(1, 4)
  adj_norm <- torch_ones(1, 1)
  output <- layer(x, adj_norm)
  expect_tensor_shape(output, c(1, 2))
})
