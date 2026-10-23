test_that("max with indices", {
  x <- torch_tensor(c(5, 6, 7, 8))
  m <- torch_max(x, dim = 1)

  expect_equal_to_r(m[[2]]$to(dtype = torch_int()), 4)

  expect_equal_to_r(
    torch_max(c(2, 1), other = c(1, 2)),
    c(2, 2)
  )
})

test_that("min with indices", {
  x <- torch_tensor(c(5, 6, 7, 8))
  m <- torch_min(x, dim = 1)

  expect_equal_to_r(m[[2]]$to(dtype = torch_int()), 1)

  expect_equal_to_r(
    torch_min(c(2, 1), other = c(1, 2)),
    c(1, 1)
  )
})

test_that("argsort", {
  x <- torch_tensor(c(3, 2, 1))
  expect_equal_to_r(torch_argsort(x), c(3, 2, 1))
  expect_equal_to_r(x$argsort(), c(3, 2, 1))

  x <- torch_tensor(c(1, 2, 3))
  expect_equal_to_r(torch_argsort(x, descending = TRUE), c(3, 2, 1))
  expect_equal_to_r(x$argsort(descending = TRUE), c(3, 2, 1))

  x <- torch_tensor(1:10)$view(c(5, 2))
  expect_equal_to_r(torch_argsort(x, dim = 1)[, 1], 1:5)
  expect_equal_to_r(x$argsort(dim = 1)[, 1], 1:5)

  expect_equal_to_r(torch_argsort(x, dim = 2)[, 1], rep(1, 5))
  expect_equal_to_r(x$argsort(dim = 2)[, 1], rep(1, 5))
})

test_that("argmax", {
  x <- torch_tensor(c(1, 2, 3))
  expect_equal_to_r(torch_argmax(x), 3)
  expect_equal_to_r(x$argmax(), 3)

  x <- torch_tensor(c(3, 2, 1))
  expect_equal_to_r(torch_argmax(x), 1)
  expect_equal_to_r(x$argmax(), 1)

  x <- torch_tensor(1:9)$reshape(c(3, 3))
  expect_equal_to_r(torch_argmax(x, dim = 2), c(3, 3, 3))
  expect_equal(torch_argmax(x, dim = 2, keepdim = TRUE)$shape, c(3, 1))
})

test_that("argmin", {
  x <- torch_tensor(c(1, 2, 3))
  expect_equal_to_r(torch_argmin(x), 1)
  expect_equal_to_r(x$argmin(), 1)

  x <- torch_tensor(c(3, 2, 1))
  expect_equal_to_r(torch_argmin(x), 3)
  expect_equal_to_r(x$argmin(), 3)

  x <- torch_tensor(1:9)$reshape(c(3, 3))
  expect_equal_to_r(torch_argmin(x, dim = 2), c(1, 1, 1))
  expect_equal(torch_argmin(x, dim = 2, keepdim = TRUE)$shape, c(3, 1))
})

test_that("sort", {
  x <- torch_tensor(sample(1e2))
  expect_equal_to_r(torch_sort(x)[[2]], order(as.integer(x)))
  expect_equal_to_r(torch_sort(x, descending = TRUE)[[2]], order(as.integer(x), decreasing = TRUE))

  expect_equal_to_r(x$sort()[[2]], order(as.integer(x)))
  expect_equal_to_r(x$sort(descending = TRUE)[[2]], order(as.integer(x), decreasing = TRUE))
})

test_that("bincount is 1 indexed", {
  x <- torch_tensor(c(1,2,3,1), dtype = torch_int64())
  out <- torch_bincount(x)
  expect_length(out, 3)
  out <- x$bincount()
  expect_length(out, 3)
  
  x <- torch_tensor(c(1,2,3,1,0), dtype = torch_int64())
  expect_error({
    out <- torch_bincount(x)  
  }, regexp =  "Indexing starts at 1 but found a 0.")
  
  
})
test_that("ignore_index is 1 indexed", {
  # `ignore_index` names a target value, and targets are 1 based in this package. It used to be
  # forwarded to libtorch unchanged while the target was converted, so it selected the class to the
  # left of the requested one, and the last class could not be ignored at all.
  logits <- torch_tensor(matrix(c(10, 0, 0, 0, 10, 0, 0, 0, 10), nrow = 3, byrow = TRUE))
  loss_for <- function(cls, ...) {
    target <- torch_tensor(rep(cls, 3L), dtype = torch_long())
    as.numeric(nnf_cross_entropy(logits, target, ...))
  }

  for (cls in 1:3) {
    # ignoring exactly the class that occurs leaves nothing to average over
    expect_true(is.nan(loss_for(cls, ignore_index = cls)))
    # while ignoring any other class leaves the loss untouched
    for (other in setdiff(1:3, cls)) {
      expect_equal(loss_for(cls, ignore_index = other), loss_for(cls), tolerance = 1e-6)
    }
  }

  # the default sentinel is not a valid target and must ignore nothing
  expect_false(is.nan(loss_for(1)))
  expect_equal(loss_for(1, ignore_index = -100), loss_for(1), tolerance = 1e-6)

  # 0 is not a valid 1 based class index
  expect_error(loss_for(1, ignore_index = 0), regexp = "1 based class index")

  # nnf_nll_loss goes through the same conversion
  log_probs <- nnf_log_softmax(logits, dim = 2)
  nll_for <- function(cls, ...) {
    target <- torch_tensor(rep(cls, 3L), dtype = torch_long())
    as.numeric(nnf_nll_loss(log_probs, target, ...))
  }
  expect_true(is.nan(nll_for(3, ignore_index = 3)))
  expect_equal(nll_for(3, ignore_index = 1), nll_for(3), tolerance = 1e-6)

  # and so does the nn_module interface
  expect_true(is.nan(as.numeric(
    nn_cross_entropy_loss(ignore_index = 2)(logits, torch_tensor(rep(2L, 3L), dtype = torch_long()))
  )))
})

test_that("backward works through max, min, sort, topk and max_pool with indices (#1185)", {
  x <- torch_tensor(matrix(c(1, 5, 3, 2, 8, 4), nrow = 2), requires_grad = TRUE)

  torch_max(x, dim = 2)[[1]]$sum()$backward()
  expect_equal_to_r(x$grad, matrix(c(0, 1, 0, 0, 1, 0), nrow = 2))

  x$grad$zero_()
  torch_min(x, dim = 2)[[1]]$sum()$backward()
  expect_equal_to_r(x$grad, matrix(c(1, 0, 0, 1, 0, 0), nrow = 2))

  for (sort in list(torch_sort, function(x, dim) x$sort(dim = dim))) {
    x$grad$zero_()
    out <- sort(x, dim = 2)
    (out[[1]] * torch_tensor(c(1, 2, 3)))$sum()$backward()
    expect_equal_to_r(out[[2]], matrix(c(1L, 2L, 2L, 3L, 3L, 1L), nrow = 2))
    expect_equal_to_r(x$grad, matrix(c(1, 3, 2, 1, 3, 2), nrow = 2))
  }

  for (topk in list(torch_topk, function(x, k) x$topk(k))) {
    x$grad$zero_()
    out <- topk(x, 2)
    out[[1]]$sum()$backward()
    expect_equal_to_r(out[[2]], matrix(c(3L, 1L, 2L, 3L), nrow = 2))
    expect_equal_to_r(x$grad, matrix(c(0, 1, 1, 0, 1, 1), nrow = 2))
  }

  y <- torch_randn(1, 1, 4, 4, requires_grad = TRUE)
  out <- nnf_max_pool2d(y, kernel_size = 2, return_indices = TRUE)
  out[[1]]$sum()$backward()
  expect_equal_to_r(y$grad$sum(), 4)
  expect_true(out[[2]]$min()$item() >= 1)
})
