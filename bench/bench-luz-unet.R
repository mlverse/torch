# Compare installed torch builds with: Rscript bench/bench-luz-unet.R
#
# Model and training loop: https://mlverse.github.io/luz/articles/examples/pets-unet.html
# Synthetic data and randomly initialized weights remove download and image
# decoding costs. UNET_BATCH_SIZE=32 matches the example's batch size; the
# smaller default makes the benchmark practical on machines with less memory.
# UNET_BATCHES, UNET_REPEATS, and UNET_THREADS control the remaining settings.
library(torch)
library(torchvision)
library(luz)

encoder <- nn_module(
  initialize = function() {
    model <- model_mobilenet_v2(pretrained = FALSE)
    self$stages <- nn_module_list(list(
      nn_identity(),
      model$features[1:2],
      model$features[3:4],
      model$features[5:7],
      model$features[8:14],
      model$features[15:18]
    ))
    for (par in self$parameters) {
      par$requires_grad_(FALSE)
    }
  },
  forward = function(x) {
    features <- list()
    for (i in seq_along(self$stages)) {
      x <- self$stages[[i]](x)
      features[[length(features) + 1]] <- x
    }
    features
  }
)

decoder_block <- nn_module(
  initialize = function(in_channels, skip_channels, out_channels) {
    self$upsample <- nn_conv_transpose2d(
      in_channels = in_channels,
      out_channels = out_channels,
      kernel_size = 2,
      stride = 2
    )
    self$activation <- nn_relu()
    self$conv <- nn_conv2d(
      in_channels = out_channels + skip_channels,
      out_channels = out_channels,
      kernel_size = 3,
      padding = "same"
    )
  },
  forward = function(x, skip) {
    x <- x %>%
      self$upsample() %>%
      self$activation()

    input <- torch_cat(list(x, skip), dim = 2)

    input %>%
      self$conv() %>%
      self$activation()
  }
)

decoder <- nn_module(
  initialize = function(
    decoder_channels = c(256, 128, 64, 32, 16),
    encoder_channels = c(16, 24, 32, 96, 320)
  ) {
    encoder_channels <- rev(encoder_channels)
    skip_channels <- c(encoder_channels[-1], 3)
    in_channels <- c(encoder_channels[1], decoder_channels)
    depth <- length(encoder_channels)

    self$blocks <- nn_module_list()
    for (i in seq_len(depth)) {
      self$blocks$append(decoder_block(
        in_channels = in_channels[i],
        skip_channels = skip_channels[i],
        out_channels = decoder_channels[i]
      ))
    }
  },
  forward = function(features) {
    features <- rev(features)
    x <- features[[1]]
    for (i in seq_along(self$blocks)) {
      x <- self$blocks[[i]](x, features[[i + 1]])
    }
    x
  }
)

model <- nn_module(
  initialize = function() {
    self$encoder <- encoder()
    self$decoder <- decoder()
    self$output <- nn_sequential(
      nn_conv2d(16, 3, 3, padding = "same")
    )
  },
  forward = function(x) {
    x %>%
      self$encoder() %>%
      self$decoder() %>%
      self$output()
  }
)

# Use identical inputs, initialization, and batch order for every trial.
batch_size <- as.integer(Sys.getenv("UNET_BATCH_SIZE", "4"))
num_batches <- as.integer(Sys.getenv("UNET_BATCHES", "4"))
repeats <- as.integer(Sys.getenv("UNET_REPEATS", "3"))
threads <- as.integer(Sys.getenv("UNET_THREADS", "4"))
torch_set_num_threads(threads)
cat(sprintf("torch=%s luz=%s torchvision=%s\n",
            packageVersion("torch"), packageVersion("luz"), packageVersion("torchvision")))
cat(sprintf("batch_size=%d batches=%d repeats=%d threads=%d\n",
            batch_size, num_batches, repeats, threads))
torch_manual_seed(20261118)
x <- torch_randn(batch_size * num_batches, 3, 224, 224)
y <- torch_randint(1, 4, c(batch_size * num_batches, 224, 224), dtype = torch_long())
dl <- dataloader(tensor_dataset(x, y), batch_size = batch_size, shuffle = FALSE)

train_once <- function() {
  torch_manual_seed(1234)
  net <- model %>% setup(optimizer = optim_adam, loss = nn_cross_entropy_loss()) %>%
    set_opt_hparams(lr = 1e-3)
  fit(net, dl, epochs = 1, verbose = FALSE)
}
# Warm up native kernels and R code before timing.
invisible(train_once())
elapsed <- numeric(repeats)
for (i in seq_len(repeats)) {
  gc()
  t <- system.time(fitted <- train_once())
  loss <- fitted$records$metrics$train[[1]]$loss
  elapsed[i] <- t[["elapsed"]]
  cat(sprintf("trial=%d elapsed=%.3f user=%.3f system=%.3f loss=%.12f\n",
              i, t[["elapsed"]], t[["user.self"]], t[["sys.self"]], loss))
}
cat(sprintf("median elapsed=%.3f seconds\n", median(elapsed)))
