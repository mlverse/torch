library(torch)

dir <- "~/Downloads/mnist"

ds <- mnist_dataset(
  dir, 
  download = TRUE, 
  transform = function(x) {
    x <- x$to(dtype = torch_float())/256
    x <- 2*(x - 0.5)
    x[newaxis,..]
  }
)
dl <- dataloader(ds, batch_size = 32, shuffle = TRUE)

generator <- nn_module(
  "generator", 
  initialize = function(latent_dim, out_channels) {
    self$main <- nn_sequential(
      nn_conv_transpose2d(latent_dim, 512, kernel_size = 4, 
                          stride = 1, padding = 0, bias = FALSE),
      nn_batch_norm2d(512),
      nn_relu(),
      nn_conv_transpose2d(512, 256, kernel_size = 4, 
                          stride = 2, padding = 1, bias = FALSE),
      nn_batch_norm2d(256),
      nn_relu(),
      nn_conv_transpose2d(256, 128, kernel_size = 4, 
                          stride = 2, padding = 1, bias = FALSE),
      nn_batch_norm2d(128),
      nn_relu(),
      nn_conv_transpose2d(128, out_channels, kernel_size = 4, 
                          stride = 2, padding = 3, bias = FALSE),
      nn_tanh()
    )
  },
  forward = function(input) {
    self$main(input)
  }
)

discriminator <- nn_module(
  "discriminator",
  initialize = function(in_channels) {
    self$main <- nn_sequential(
      nn_conv2d(in_channels, 16, kernel_size = 4, stride = 2, padding = 1, bias = FALSE),
      nn_leaky_relu(0.2, inplace = TRUE),
      nn_conv2d(16, 32, kernel_size = 4, stride = 2, padding = 1, bias = FALSE),
      nn_batch_norm2d(32),
      nn_leaky_relu(0.2, inplace = TRUE),
      nn_conv2d(32, 64, kernel_size = 4, stride = 2, padding = 1, bias = FALSE),
      nn_batch_norm2d(64),
      nn_leaky_relu(0.2, inplace = TRUE),
      nn_conv2d(64, 128, kernel_size = 4, stride = 2, padding = 1, bias = FALSE),
      nn_leaky_relu(0.2, inplace = TRUE)
    )
    self$linear <- nn_linear(128, 1)
    self$sigmoid <- nn_sigmoid()
  },
  forward = function(input) {
    x <- self$main(input)
    x <- torch_flatten(x, start_dim = 2)
    x <- self$linear(x)
    self$sigmoid(x)
  }
)

plot_gen <- function(noise) {
  img <- G(noise)
  img <- img$cpu()
  img <- img[1,1,,,newaxis]/2 + 0.5
  img <- torch_stack(list(img, img, img), dim = 2)[..,1]
  img <- as.raster(as_array(img))
  plot(img)
}

device <- torch_device(ifelse(cuda_is_available(),  "cuda", "cpu"))

G <- generator(latent_dim = 100, out_channels = 1)
D <- discriminator(in_channels = 1)

init_weights <- function(m) {
  if (grepl("conv", m$.classes[[1]])) {
    nn_init_normal_(m$weight$data(), 0.0, 0.02)
  } else if (grepl("batch_norm", m$.classes[[1]])) {
    nn_init_normal_(m$weight$data(), 1.0, 0.02)
    nn_init_constant_(m$bias$data(), 0)
  } 
}

G[[1]]$apply(init_weights)
D[[1]]$apply(init_weights)

G$to(device = device)
D$to(device = device)

G_optimizer <- optim_adam(G$parameters, lr = 2 * 1e-4, betas = c(0.5, 0.999))
D_optimizer <- optim_adam(D$parameters, lr = 2 * 1e-4, betas = c(0.5, 0.999))

fixed_noise <- torch_randn(1, 100, 1, 1, device = device)

loss <- nn_bce_loss()

for (epoch in 1:10) {
  
  pb <- progress::progress_bar$new(
    total = length(dl), 
    format = "[:bar] :eta Loss D: :lossd Loss G: :lossg"
  )
  lossg <- c()
  lossd <- c()
  
  for (b in enumerate(dl)) {
    
    y_real <- torch_ones(32, device = device)
    y_fake <- torch_zeros(32, device = device)
    
    noise <- torch_randn(32, 100, 1, 1, device = device)
    fake <- G(noise)
    
    img <- b[[1]]$to(device = device)
    
    # train the discriminator ---
    D_loss <- loss(D(img), y_real) + loss(D(fake$detach()), y_fake)
    
    D_optimizer$zero_grad()
    D_loss$backward()
    D_optimizer$step()
    
    # train the generator ---
    
    G_loss <- loss(D(fake), y_real)
    
    G_optimizer$zero_grad()
    G_loss$backward()
    G_optimizer$step()
    
    lossd <- c(lossd, D_loss$item())
    lossg <- c(lossg, G_loss$item())
    pb$tick(tokens = list(lossd = mean(lossd), lossg = mean(lossg)))
  }
  plot_gen(fixed_noise)
  
  cat(sprintf("Epoch %d - Loss D: %3f Loss G: %3f\n", epoch, mean(lossd), mean(lossg)))
}
