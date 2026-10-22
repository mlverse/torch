context("nn-multihead_attention")

test_that("nn_multihead_attention", {
  t1 <- torch_randn(5, 8, 32)
  t2 <- torch_randn(5, 8, 32)
  t3 <- torch_randn(5, 8, 32)
  attn <- nn_multihead_attention(32, 8)

  # q,k,v all the same:
  out <- attn(t1, t1, t1)

  expect_identical(out[[1]]$size(), c(5L, 8L, 32L))
  expect_identical(out[[2]]$size(), c(8L, 5L, 5L))

  # unaveraged attention weights
  out <- attn(t1, t1, t1, avg_weights = FALSE)

  expect_identical(out[[1]]$size(), c(5L, 8L, 32L))
  expect_identical(out[[2]]$size(), c(8L, 8L, 5L, 5L))

  # q different from k,v:
  out <- attn(t1, t2, t2)

  expect_identical(out[[1]]$size(), c(5L, 8L, 32L))
  expect_identical(out[[2]]$size(), c(8L, 5L, 5L))

  # q,k,v all different
  out <- attn(t1, t2, t3)

  expect_identical(out[[1]]$size(), c(5L, 8L, 32L))
  expect_identical(out[[2]]$size(), c(8L, 5L, 5L))

  t2 <- torch_ones(c(5, 5)) - torch_tril(torch_ones(c(5, 5)))
  t2 <- t2$to(torch_bool())
  t3 <- torch_bernoulli(torch_ones(c(8, 5)) * 0.5)$to(dtype = "bool")
  out2 <- attn(t1, t1, t1, attn_mask = t2, key_padding_mask = t3)

  expect_identical(out2[[1]]$size(), c(5L, 8L, 32L))
  expect_identical(out2[[2]]$size(), c(8L, 5L, 5L))

})

test_that("nn_multihead_attention gives the same output with need_weights = FALSE (scaled dot product attention)", {
  torch_manual_seed(1)
  L <- 5; S <- 7; N <- 3; E <- 16; H <- 4
  q <- torch_randn(L, N, E)
  kv <- torch_randn(S, N, E)
  v2 <- torch_randn(S, N, E)

  # masks for a source sequence of length `S`
  make_masks <- function(S) {
    bool_mask <- torch_rand(L, S) > 0.7
    bool_mask[, 1] <- FALSE # every query attends to at least one key
    list(
      none = list(),
      attn_bool = list(attn_mask = bool_mask),
      attn_float = list(attn_mask = torch_randn(L, S)),
      attn_3d = list(attn_mask = torch_randn(N * H, L, S)),
      key_padding_bool = list(
        key_padding_mask = torch_tensor(matrix(rep_len(c(FALSE, TRUE), S), N, S, byrow = TRUE))$to(dtype = torch_bool())
      ),
      key_padding_float = list(key_padding_mask = torch_randn(N, S)),
      both = list(attn_mask = torch_randn(L, S), key_padding_mask = torch_randn(N, S))
    )
  }
  masks_self <- make_masks(L)
  masks <- make_masks(S)

  compare <- function(attn, query, key, value, args = list()) {
    attn$eval()
    expected <- do.call(attn, c(list(query, key, value), args))[[1]]
    out <- do.call(attn, c(list(query, key, value, need_weights = FALSE), args))
    expect_null(out[[2]])
    expect_equal_to_tensor(out[[1]], expected, tolerance = 1e-5)
  }

  attn <- nn_multihead_attention(E, H)
  for (args in masks_self) {
    compare(attn, q, q, q, args) # self-attention
  }
  for (args in masks) {
    compare(attn, q, kv, kv, args) # key and value are the same
    compare(attn, q, kv, v2, args) # all different
  }

  compare(nn_multihead_attention(E, H, bias = FALSE), q, kv, v2)
  compare(nn_multihead_attention(E, H, add_bias_kv = TRUE), q, kv, kv, masks$attn_float)
  compare(nn_multihead_attention(E, H, add_zero_attn = TRUE), q, kv, kv, masks$key_padding_float)
  compare(nn_multihead_attention(E, H, kdim = 6, vdim = 10), q, torch_randn(S, N, 6), torch_randn(S, N, 10),
    masks$both)

  attn <- nn_multihead_attention(E, H, batch_first = TRUE)
  qb <- torch_randn(N, L, E)
  compare(attn, qb, qb, qb, masks_self$both)
  compare(attn, qb, torch_randn(N, S, E), torch_randn(N, S, E), masks$attn_bool)
})

test_that("nn_multihead_attention detects self-attention by identity, with the same result for equal tensors", {
  torch_manual_seed(1)
  attn <- nn_multihead_attention(8, 2)
  x <- torch_randn(4, 2, 8)
  y <- x$clone()
  expect_equal_to_tensor(attn(x, x, x)[[1]], attn(x, y, y)[[1]], tolerance = 1e-6)
  expect_equal_to_tensor(attn(x, x, x)[[1]], attn(x, y, x)[[1]], tolerance = 1e-6)
  expect_equal_to_tensor(attn(x, x, x)[[2]], attn(x, y, y$clone())[[2]], tolerance = 1e-6)
})

test_that("nn_multihead_attention works with add_zero_attn", {
  torch_manual_seed(1)
  attn <- nn_multihead_attention(8, 2, add_zero_attn = TRUE)
  x <- torch_randn(4, 2, 8)
  out <- attn(x, x, x)
  expect_identical(out[[1]]$size(), c(4L, 2L, 8L))
  # one more key, the zero key
  expect_identical(out[[2]]$size(), c(2L, 4L, 5L))
})

test_that("nn_multihead_attention with need_weights = FALSE supports gradients and dropout", {
  torch_manual_seed(1)
  attn <- nn_multihead_attention(8, 2, dropout = 0.5)
  x <- torch_randn(4, 2, 8, requires_grad = TRUE)
  out <- attn(x, x, x, need_weights = FALSE)[[1]]
  out$sum()$backward()
  expect_true(all(is.finite(as_array(x$grad))))
  expect_true(all(is.finite(as_array(attn$in_proj_weight$grad))))
})

test_that("nn_multihead_attention supports the is_causal hint", {
  torch_manual_seed(1)
  attn <- nn_multihead_attention(16, 4)
  attn$eval()
  x <- torch_randn(6, 3, 16)
  causal <- torch_ones(6, 6, dtype = torch_bool())$triu(diagonal = 1)
  expected <- attn(x, x, x, attn_mask = causal)[[1]]
  expect_equal_to_tensor(attn(x, x, x, attn_mask = causal, need_weights = FALSE, is_causal = TRUE)[[1]], expected,
    tolerance = 1e-5)
  expect_equal_to_tensor(attn(x, x, x, attn_mask = causal, is_causal = TRUE)[[1]], expected, tolerance = 1e-5)
  kpm <- torch_tensor(matrix(c(FALSE, FALSE, FALSE, FALSE, TRUE, TRUE), 3, 6, byrow = TRUE))
  expect_equal_to_tensor(
    attn(x, x, x, attn_mask = causal, key_padding_mask = kpm, need_weights = FALSE, is_causal = TRUE)[[1]],
    attn(x, x, x, attn_mask = causal, key_padding_mask = kpm)[[1]],
    tolerance = 1e-5
  )
  expect_error(attn(x, x, x, is_causal = TRUE), "Need attn_mask")
})

test_that("nn_multihead_attention treats integer masks as boolean masks", {
  torch_manual_seed(1)
  attn <- nn_multihead_attention(8, 2)
  x <- torch_randn(4, 2, 8)
  mask <- torch_rand(4, 4) > 0.6
  mask[, 1] <- FALSE
  kpm <- torch_tensor(matrix(c(FALSE, TRUE, FALSE, FALSE), 2, 4, byrow = TRUE))
  for (need_weights in c(TRUE, FALSE)) {
    expect_equal_to_tensor(
      attn(x, x, x, attn_mask = mask$to(dtype = torch_uint8()), key_padding_mask = kpm$to(dtype = torch_long()),
        need_weights = need_weights)[[1]],
      attn(x, x, x, attn_mask = mask, key_padding_mask = kpm)[[1]],
      tolerance = 1e-6
    )
  }
})

test_that("nn_multihead_attention accepts non-contiguous masks and checks their shapes", {
  torch_manual_seed(1)
  attn <- nn_multihead_attention(8, 2)
  x <- torch_randn(4, 3, 8)
  kpm <- torch_randn(4, 3)$t()
  attn_mask <- torch_randn(2 * 3, 4, 4)$transpose(2, 3)
  for (need_weights in c(TRUE, FALSE)) {
    expect_equal_to_tensor(
      attn(x, x, x, attn_mask = attn_mask, key_padding_mask = kpm, need_weights = need_weights)[[1]],
      attn(x, x, x, attn_mask = attn_mask$contiguous(), key_padding_mask = kpm$contiguous())[[1]],
      tolerance = 1e-6
    )
    expect_error(attn(x, x, x, attn_mask = torch_randn(5, 4), need_weights = need_weights), "should be")
    expect_error(attn(x, x, x, key_padding_mask = torch_randn(3, 5), need_weights = need_weights), "key_padding_mask")
  }
})

test_that("nn_multihead_attention passes avg_weights on with kdim and vdim", {
  attn <- nn_multihead_attention(8, 2, kdim = 5, vdim = 6)
  out <- attn(torch_randn(4, 2, 8), torch_randn(3, 2, 5), torch_randn(3, 2, 6), avg_weights = FALSE)
  expect_identical(out[[2]]$size(), c(2L, 2L, 4L, 3L))
})

test_that("nn_multihead_attention uses the native implementation for inference, with the same results", {
  torch_manual_seed(1)
  attn <- nn_multihead_attention(16, 4, batch_first = TRUE)
  attn$eval()
  x <- torch_randn(3, 5, 16)
  mask <- torch_rand(5, 5) > 0.7
  mask[, 1] <- FALSE
  kpm <- torch_tensor(matrix(c(FALSE, FALSE, FALSE, FALSE, TRUE), 3, 5, byrow = TRUE))
  expect_true(with_no_grad(attn$can_use_native_attention(x, x, x, mask, kpm)))
  expect_false(attn$can_use_native_attention(x, x, x)) # the parameters require gradients
  # the native implementation treats masks as boolean
  expect_false(with_no_grad(attn$can_use_native_attention(x, x, x, torch_randn(5, 5))))
  for (need_weights in c(TRUE, FALSE)) {
    native <- with_no_grad(attn(x, x, x, attn_mask = mask, key_padding_mask = kpm, need_weights = need_weights))
    expected <- attn(x, x, x, attn_mask = mask, key_padding_mask = kpm, need_weights = need_weights)
    expect_equal_to_tensor(native[[1]], expected[[1]]$detach(), tolerance = 1e-5)
    if (need_weights) {
      expect_equal_to_tensor(native[[2]], expected[[2]]$detach(), tolerance = 1e-5)
    } else {
      expect_null(native[[2]])
    }
  }
})

test_that("nn_multihead_attention returns contiguous outputs with batch_first and need_weights = FALSE", {
  attn <- nn_multihead_attention(8, 2, batch_first = TRUE)
  x <- torch_randn(3, 4, 8)
  expect_true(attn(x, x, x, need_weights = FALSE)[[1]]$is_contiguous())
})
