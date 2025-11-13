# Automatic Mixed Precision

``` r
library(torch)
torch_manual_seed(1)
```

Automatic Mixed Precision (AMP) is a technique that enables faster
training of deep learning models while maintaining model accuracy by
using a combination of single-precision (FP32) and half-precision (FP16)
floating-point formats.

Modern [NVIDIA
GPU’s](https://developer.nvidia.com/automatic-mixed-precision) have
improved support for AMP and torch can benefit of it with minimal code
modifications.

In torch, the AMP implementation involves 2 steps:

1.  Allow the model to use different implementations of operations so it
    can use half-precision.
2.  Using loss scaling to preserve small gradient values that might be
    too small due to using half-precision.

The first step, can also be used to speedup inference pipelines.

## Example

We will define a model and some random training data to showcase how to
enabled AMP with torch:

``` r
batch_size <- 512 # Try, for example, 128, 256, 513.
in_size <- 4096
out_size <- 4096
num_layers <- 3
num_batches <- 50
epochs <- 3

# Creates data in default precision.
# The same data is used for both default and mixed precision trials below.
# You don't need to manually change inputs' dtype when enabling mixed precision.
data <- lapply(1:num_batches, function(x) torch_randn(batch_size, in_size, device="cuda"))
targets <- lapply(1:num_batches, function(x) torch_randn(batch_size, out_size, device="cuda"))
```

Now let’s define the model:

``` r
make_model <- function(in_size, out_size, num_layers) {
  layers <- list()
  for (i in seq_len(num_layers-1)) {
    layers <- c(layers, list(nn_linear(in_size, in_size), nn_relu()))
  }
  layers <- c(layers, list(nn_linear(in_size, out_size)))
  nn_sequential(!!!layers)$cuda()
}
```

To train the model without mixed precision we can do:

``` r
loss_fn <- nn_mse_loss()$cuda()
net <- make_model(in_size, out_size, num_layers)
opt <- optim_sgd(net$parameters, lr=0.1)

for (epoch in seq_len(epochs)) {
  for (i in seq_along(data)) {
    
    output <- net(data[[i]])
    loss <- loss_fn(output, targets[[i]])
  
    loss$backward()
    opt$step()
    opt$zero_grad()
  }
}
```

To enabled step 1. of mixed precision, ie, allowing the model to run
operations with half precision when possible, one simply run model
computations (including the loss computation) inside a `with_autocast`
context.

``` r
loss_fn <- nn_mse_loss()$cuda()
net <- make_model(in_size, out_size, num_layers)
opt <- optim_sgd(net$parameters, lr=0.1)

for (epoch in seq_len(epochs)) {
  for (i in seq_along(data)) {
    with_autocast(device_type = "cuda", {
      output <- net(data[[i]])
      loss <- loss_fn(output, targets[[i]])  
    })
    
    loss$backward()
    opt$step()
    opt$zero_grad()
  }
}
```

To additionally enable gradient scaling we will now introduce the
[`cuda_amp_grad_scaler()`](https://torch.mlverse.org/docs/dev/reference/cuda_amp_grad_scaler.md)
object and use it scale the loss before calling `backward()` and also
use it to wrap calls to the optimizer, so it can ‘unscale’ the gradients
before actually updating the weights. The training loop is now
implemented as:

``` r
loss_fn <- nn_mse_loss()$cuda()
net <- make_model(in_size, out_size, num_layers)
opt <- optim_sgd(net$parameters, lr=0.1)
scaler <- cuda_amp_grad_scaler()

for (epoch in seq_len(epochs)) {
  for (i in seq_along(data)) {
    with_autocast(device_type = "cuda", {
      output <- net(data[[i]])
      loss <- loss_fn(output, targets[[i]])  
    })
    
    scaler$scale(loss)$backward()
    scaler$step(opt)
    scaler$update()
    opt$zero_grad()
  }
}
```

## Benchmark

We now write a simple function that allows us to quickly switch between
feature so we can benchmark AMP:

``` r
run <- function(autocast, scale) {
  loss_fn <- nn_mse_loss()$cuda()
  net <- make_model(in_size, out_size, num_layers)
  opt <- optim_sgd(net$parameters, lr=0.1)
  scaler <- cuda_amp_grad_scaler(enabled = scale)
  
  for (epoch in seq_len(epochs)) {
    for (i in seq_along(data)) {
      with_autocast(enabled = autocast, device_type = "cuda", {
        output <- net(data[[i]])
        loss <- loss_fn(output, targets[[i]])  
      })
      
      scaler$scale(loss)$backward()
      scaler$step(opt)
      scaler$update()
      opt$zero_grad()
    }
  }
  loss$item()
}


system.time({run(autocast = FALSE, scale = FALSE)})
#>    user  system elapsed 
#>   3.967   1.644   5.621
system.time({run(autocast = TRUE, scale = FALSE)})
#>    user  system elapsed 
#>   1.857   0.591   2.462
system.time({run(autocast = TRUE, scale = TRUE)})
#>    user  system elapsed 
#>   3.434   0.114   3.550
```
