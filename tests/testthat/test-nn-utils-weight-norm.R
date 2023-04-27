test_that("nn_utils_weight_norm", {
  in_feats = 20
  out_feats = 40
  weight_norm = nn_utils_weight_norm$new('weight', 2)
  lin = nn_linear(in_feats, out_feats)
  weight_norm$apply(lin) 
  
  lin_weight_before = as.array(lin$weight)
  weight_norm$recompute(lin)
  lin_weight_after = as.array(lin$weight)
  
  expect_equal(lin_weight_before, lin_weight_after)
  
  expect_tensor(lin$weight_g)
  expect_tensor_shape(lin$weight_g, c(1, in_feats))
  
  expect_tensor(lin$weight_v)
  expect_tensor_shape(lin$weight_v, c(out_feats, in_feats))
  
  weight_norm$call(lin)
  lin_weight_call = as.array(lin$weight)
  
  expect_equal(lin_weight_call, lin_weight_before)
  expect_equal(capture.output(lin$weight$grad_fn), "WeightNormInterfaceBackward0")
  
  weight_norm$remove(lin)
  lin_weight_remove = as.array(lin$weight)
  expect_equal(lin_weight_remove, lin_weight_before)
  expect_null(lin$weight$grad_fn)
  expect_null(lin$parameters$weight_v)
  expect_null(lin$parameters$weight_g)
  expect_true(lin$weight$requires_grad)
})

