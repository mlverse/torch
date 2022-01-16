context("nn-rnn")

test_that("rnn nonlinearity", {
  rnn <- nn_rnn(1, 10)
  expect_equal(rnn$nonlinearity, "tanh")

  rnn <- nn_rnn(1, 10, nonlinearity = "relu")
  expect_equal(rnn$nonlinearity, "relu")

  expect_error(
    rnn <- nn_rnn(1, 10, nonlinearity = "garbage"),
    class = "value_error"
  )
})

test_that("rnn dropout", {
  for (p in c(0., .276, .731, 1)) {
    for (train in c(TRUE, FALSE)) {
      rnn <- nn_rnn(10, 1000, 2, bias = FALSE, dropout = p, nonlinearity = "relu")

      with_no_grad({
        rnn$weight_ih_l1$fill_(1)
        rnn$weight_hh_l1$fill_(1)
        rnn$weight_ih_l2$fill_(1)
        rnn$weight_hh_l2$fill_(1)
      })

      if (train) {
        rnn$train()
      } else {
        rnn$eval()
      }

      input <- torch_ones(1, 1, 10)
      hx <- torch_zeros(2, 1, 1000)

      out <- rnn(input, hx)
      output <- out[[1]]
      hy <- out[[2]]


      expect_equal_to_tensor(output$min(), output$max(), tolerance = 1e-2)

      output_val <- output[1, 1, 1]

      if (p == 0 || !train) {
        expect_equal_to_r(output_val, 10000)
      } else if (p == 1) {
        expect_equal_to_r(output_val, 0)
      } else {
        expect_equal_to_r(output_val > 8000, TRUE)
        expect_equal_to_r(output_val < 12000, TRUE)
      }

      expect_equal_to_tensor(hy[1, , ]$min(), hy[1, , ]$max(), tolerance = 1e-2)
      expect_equal_to_tensor(hy[2, , ]$min(), hy[2, , ]$max(), tolerance = 1e-2)
      expect_equal_to_r(hy[1, 1, 1], 10)
      expect_equal_to_tensor(hy[2, 1, 1], output_val, tolerance = 1e-2)
    }
  }
})

test_that("rnn packed sequence", {
  x <- torch_tensor(rbind(
    c(1, 2, 0, 0),
    c(1, 2, 3, 0),
    c(1, 2, 3, 4)
  ), dtype = torch_float())
  x <- x[, , newaxis]
  lens <- torch_tensor(c(2, 3, 4), dtype = torch_long())

  p <- nn_utils_rnn_pack_padded_sequence(x, lens,
    batch_first = TRUE,
    enforce_sorted = FALSE
  )

  rnn <- nn_rnn(1, 4, nonlinearity = "relu")
  out <- rnn(p)

  unpack <- nn_utils_rnn_pad_packed_sequence(out[[1]])
  expect_tensor_shape(unpack[[1]], c(4, 3, 4))
  expect_equal_to_r(unpack[[2]]$to(dtype = torch_int()), c(2, 3, 4))
})

test_that("lstm", {
  lstm <- nn_lstm(10, 5)
  expect_equal(lstm$mode, "LSTM")

  input <- torch_ones(1, 1, 10)
  o <- lstm(input)
  expect_length(o, 2)

  expect_tensor_shape(o[[1]], c(1, 1, 5))
  expect_tensor_shape(o[[2]][[1]], c(1, 1, 5))
  expect_tensor_shape(o[[2]][[2]], c(1, 1, 5))

  expect_tensor_shape(lstm$weight_ih_l1, c(20, 10))
  expect_tensor_shape(lstm$weight_hh_l1, c(20, 5))
  expect_tensor_shape(lstm$bias_ih_l1, c(20))
  expect_tensor_shape(lstm$bias_hh_l1, c(20))

  expect_length(lstm$parameters, 4)

  with_no_grad({
    lstm$weight_ih_l1$fill_(1)
    lstm$weight_hh_l1$fill_(1)
    lstm$bias_ih_l1$fill_(1)
    lstm$bias_hh_l1$fill_(1)
  })

  z <- lstm(input)

  expect_equal_to_tensor(z[[1]], torch_ones(1, 1, 5) * 0.7615868, tolerance = 1e-5)
  expect_equal_to_tensor(z[[2]][[1]], torch_ones(1, 1, 5) * 0.7615868, tolerance = 1e-5)
  expect_equal_to_tensor(z[[2]][[2]], torch_ones(1, 1, 5), tolerance = 1e-5)

  lstm <- nn_lstm(10, 5, bias = FALSE)

  expect_tensor_shape(lstm$weight_ih_l1, c(20, 10))
  expect_tensor_shape(lstm$weight_hh_l1, c(20, 5))
  expect_null(lstm$bias_ih_l1)
  expect_null(lstm$bias_hh_l1, NULL)

  with_no_grad({
    lstm$weight_ih_l1$fill_(1)
    lstm$weight_hh_l1$fill_(1)
  })

  z <- lstm(input)

  expect_equal_to_tensor(z[[1]], torch_ones(1, 1, 5) * 0.7615405, tolerance = 1e-5)
  expect_equal_to_tensor(z[[2]][[1]], torch_ones(1, 1, 5) * 0.7615405, tolerance = 1e-4)
  expect_equal_to_tensor(z[[2]][[2]], torch_ones(1, 1, 5), tolerance = 1e-4)

  lstm <- nn_lstm(10, 5, num_layers = 2)
  expect_length(lstm$parameters, 8)
  lstm <- nn_lstm(10, 5, num_layers = 3)
  expect_length(lstm$parameters, 12)

  with_no_grad({
    for (p in lstm$parameters) {
      p$fill_(1)
    }
  })

  z <- lstm(input)
  expect_equal_to_tensor(z[[1]], torch_ones(1, 1, 5) * 0.7580, tolerance = 1e-4)

  expect_equal_to_tensor(z[[2]][[1]][1, , ], torch_ones(1, 5) * 0.7616, tolerance = 1e-4)
  expect_equal_to_tensor(z[[2]][[1]][2, , ], torch_ones(1, 5) * 0.7580, tolerance = 1e-4)
  expect_equal_to_tensor(z[[2]][[1]][3, , ], torch_ones(1, 5) * 0.7580, tolerance = 1e-4)

  expect_equal_to_tensor(z[[2]][[2]][1, , ], torch_ones(1, 5), tolerance = 1e-4)
  expect_equal_to_tensor(z[[2]][[2]][2, , ], torch_ones(1, 5) * 0.9970, tolerance = 1e-4)
  expect_equal_to_tensor(z[[2]][[2]][3, , ], torch_ones(1, 5) * 0.9969, tolerance = 1e-4)
})

test_that("gru", {
  gru <- nn_gru(10, 5)
  expect_equal(gru$mode, "GRU")

  input <- torch_ones(1, 1, 10)
  o <- gru(input)
  expect_length(o, 2)

  expect_tensor_shape(o[[1]], c(1, 1, 5))
  expect_tensor_shape(o[[2]], c(1, 1, 5))

  expect_tensor_shape(gru$weight_ih_l1, c(15, 10))
  expect_tensor_shape(gru$weight_hh_l1, c(15, 5))
  expect_tensor_shape(gru$bias_ih_l1, c(15))
  expect_tensor_shape(gru$bias_hh_l1, c(15))

  expect_length(gru$parameters, 4)

  with_no_grad({
    gru$weight_ih_l1$fill_(1)
    gru$weight_hh_l1$fill_(1)
    gru$bias_ih_l1$fill_(1)
    gru$bias_hh_l1$fill_(1)
  })

  z <- gru(input)

  expect_equal_to_tensor(z[[1]], torch_ones(1, 1, 5) * 6.1989e-06, tolerance = 1e-5)
  expect_equal_to_tensor(z[[2]], torch_ones(1, 1, 5) * 6.1989e-06, tolerance = 1e-5)

  gru <- nn_gru(10, 5, bias = FALSE)

  expect_tensor_shape(gru$weight_ih_l1, c(15, 10))
  expect_tensor_shape(gru$weight_hh_l1, c(15, 5))
  expect_null(gru$bias_ih_l1)
  expect_null(gru$bias_hh_l1, NULL)

  with_no_grad({
    gru$weight_ih_l1$fill_(1)
    gru$weight_hh_l1$fill_(1)
  })

  z <- gru(input)

  expect_equal_to_tensor(z[[1]], torch_ones(1, 1, 5) * 4.5419e-05, tolerance = 1e-5)
  expect_equal_to_tensor(z[[2]], torch_ones(1, 1, 5) * 4.5419e-05, tolerance = 1e-4)

  gru <- nn_gru(10, 5, num_layers = 2)
  expect_length(gru$parameters, 8)
  gru <- nn_gru(10, 5, num_layers = 3)
  expect_length(gru$parameters, 12)

  with_no_grad({
    for (p in gru$parameters) {
      p$fill_(1)
    }
  })

  z <- gru(input)
  expect_equal_to_tensor(z[[1]], torch_ones(1, 1, 5) * 0.0702, tolerance = 1e-4)

  expect_equal_to_tensor(z[[2]][1, , ], torch_ones(1, 5) * 6.1989e-06, tolerance = 1e-4)
  expect_equal_to_tensor(z[[2]][2, , ], torch_ones(1, 5) * 1.1378e-01, tolerance = 1e-4)
  expect_equal_to_tensor(z[[2]][3, , ], torch_ones(1, 5) * 7.0209e-02, tolerance = 1e-4)
})

test_that("rnn gpu", {
  skip_if_cuda_not_available()

  rnn <- nn_rnn(10, 1)
  rnn$to(device = "cuda")

  input <- torch_ones(1, 1, 10, device = "cuda")

  expect_message(out <- rnn(input), regexp = NA)

  expect_length(out, 2)
  expect_tensor_shape(out[[1]], c(1, 1, 1))
  expect_tensor_shape(out[[2]], c(1, 1, 1))
})

test_that("GRU on the GPU keeps its parameters", {
  skip_if_cuda_not_available()

  model <- nn_module(
    initialize = function(input_size, hidden_size) {
      self$rnn <- nn_gru(
        input_size = input_size,
        hidden_size = hidden_size,
        batch_first = TRUE
      )
      self$output <- nn_linear(hidden_size, 1)
    },
    forward = function(x) {
      # list of [output, hidden]
      # we are interested in the final timestep only, so we can directly use [[2]]
      # but we want to remove the un-needed singleton dimension on the left
      x <- self$rnn(x)[[2]]$squeeze(1)
      x %>% self$output()
    }
  )
  m <- model(1, 64)
  e_pars <- names(m$parameters)
  m$cuda()
  r_pars <- names(m$parameters)

  expect_equal(r_pars, e_pars)
})

test_that("lstm and gru works with packed sequences", {
  # regression test for https://github.com/mlverse/torch/issues/499

  x <- torch_tensor(rbind(
    c(1, 2, 0, 0),
    c(1, 2, 3, 0),
    c(1, 2, 3, 4)
  ), dtype = torch_float())
  x <- x[, , newaxis]
  lens <- torch_tensor(c(2, 3, 4), dtype = torch_long())

  p <- nn_utils_rnn_pack_padded_sequence(x, lens,
    batch_first = TRUE,
    enforce_sorted = FALSE
  )

  rnn <- nn_lstm(1, 4)
  out <- rnn(p)

  unpack <- nn_utils_rnn_pad_packed_sequence(out[[1]])
  expect_tensor_shape(unpack[[1]], c(4, 3, 4))

  rnn <- nn_gru(1, 4)
  out <- rnn(p)

  unpack <- nn_utils_rnn_pad_packed_sequence(out[[1]])
  expect_tensor_shape(unpack[[1]], c(4, 3, 4))
})

test_that("gru can be traced", {
  x <- nn_gru(10, 10)
  tr <- jit_trace(x, torch_randn(10, 10, 10))

  v <- torch_randn(10, 10, 10)
  expect_equal_to_tensor(
    x(v)[[1]],
    tr(v)[[1]]
  )
})
