check_log_prob <- function(distribution, expect_fn) {
  s <- distribution$sample()
  log_probs <- distribution$log_prob(s)
  log_probs_data_flat <- log_probs$view(-1)
  s_data_flat <- s$view(c(length(log_probs_data_flat), -1))

  for (i in seq_along(s_data_flat)) {
    val <- s_data_flat[i]
    log_prob <- log_probs_data_flat[i]
    expect_fn(i, val$squeeze(), log_prob)
  }
}

check_enumerate_support <- function(distr_cls, examples) {
  for (i in seq_along(examples)) {
    params <- examples[[i]][[1]]
    params <- Map(torch_tensor, params)
    expected <- examples[[i]][[2]]

    # TODO: consider ignoring arg types in this expect_equal (suggested in PyTorch)
    expected <- torch_tensor(expected)
    d <- do.call(distr_cls, params)
    actual <- d$enumerate_support(expand = FALSE)
    expect_equal(actual, expected)

    actual <- d$enumerate_support(expand = TRUE)
    expected_with_expand <- expected$expand(c(-1, d$batch_shape, d$event_shape))
    expect_equal(actual, expected_with_expand)
  }
}
