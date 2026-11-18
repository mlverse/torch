test_that("optim_adamw", {
  expect_optim_works(optim_adamw, list(lr = 0.1))
  expect_optim_works(optim_adamw, list(lr = 0.1, weight_decay = 1e-5))
  expect_optim_works(optim_adamw, list(lr = 0.1, weight_decay = 1e-5, amsgrad = TRUE))
  expect_state_is_updated(optim_adamw)
})

test_that("AdamW keeps a separate step count for each parameter", {
  for (amsgrad in c(FALSE, TRUE)) {
    x <- torch_tensor(1, requires_grad = TRUE)
    y <- torch_tensor(2, requires_grad = TRUE)
    opt <- optim_adamw(list(x, y), amsgrad = amsgrad)
    x$grad <- torch_tensor(1)
    opt$step()
    opt$step()
    y$grad <- torch_tensor(1)
    opt$step()

    saved <- opt$state_dict()
    expect_equal(as.numeric(saved$state[["1"]]$step), 3)
    expect_equal(as.numeric(saved$state[["2"]]$step), 1)
    if (amsgrad) {
      expect_true(is_torch_tensor(saved$state[["1"]]$max_exp_avg_sqs))
    }
  }
})
