context("nn-multihead_attention")

test_that("nn_multihead_attention", {
    t1 <- torch_randn(5, 8, 32)
    attn <- nn_multihead_attention(32, 8)
    out <- attn(t1, t1, t1)

    expect_identical(out[[1]]$size(), c(5L, 8L, 32L))
    expect_identical(out[[2]]$size(), c(8L, 5L, 5L))

    t2 <- torch_ones(c(5, 5)) - torch_tril(torch_ones(c(5, 5)))
    t3 <- torch_bernoulli(torch_ones(c(8, 5)) * 0.5)
    out2 <- attn(t1, t1, t1, attn_mask = t2, key_padding_mask = t3)

    expect_identical(out2[[1]]$size(), c(5L, 8L, 32L))
    expect_identical(out2[[2]]$size(), c(8L, 5L, 5L))
})