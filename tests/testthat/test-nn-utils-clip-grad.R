test_that("clip_grad_norm_", {
  compute_norm <- function(parameters, norm_type) {
    if (is.finite(norm_type)) {
      total_norm <- 0
      for (p in parameters) {
        total_norm <- total_norm + p$grad$data()$abs()$pow(norm_type)$sum()
      }
      total_norm^(1 / norm_type)
    } else {
      torch_tensor(max(sapply(parameters, function(p) p$grad$data()$abs()$max()$item())))
    }
  }

  grads <- list(
    torch_arange(start = 1, end = 100)$view(c(10, 10)),
    torch_ones(10)$div(1000)
  )

  l <- nn_linear(10, 10)
  max_norm <- 2

  for (norm_type in c(0.5, 1.5, 2, 4, Inf)) {
    for (i in seq_along(l$parameters)) {
      l$parameters[[i]]$set_grad_(grads[[i]]$clone())
    }

    norm_before <- compute_norm(l$parameters, norm_type)
    norm <- nn_utils_clip_grad_norm_(l$parameters, max_norm, norm_type)
    norm_after <- compute_norm(l$parameters, norm_type)
    expect_equal_to_tensor(norm_before, norm, tolerance = 1e-2)
    expect_equal_to_tensor(norm_after, torch_tensor(max_norm), tolerance = 1e-2)
    expect_equal_to_r(norm_after < norm_before, TRUE)
  }
})

test_that("clip_grad_value_", {
  grads <- list(
    torch_arange(start = 1, end = 100)$view(c(10, 10)),
    torch_ones(10)$div(1000)
  )

  l <- nn_linear(10, 10)
  max_norm <- 2

  for (i in seq_along(l$parameters)) {
    l$parameters[[i]]$set_grad_(grads[[i]])
  }

  nn_utils_clip_grad_value_(l$parameters, 2.5)

  for (p in l$parameters) {
    expect_equal_to_r(p$grad$max() <= 2.5, TRUE)
  }
})
