---
title: "Guess the correlation: regression with torch end-to-end"
weight: 1
description: | 
  First torch example
---

# Get the packages

To use `torch`, you first need to install it. Get the CRAN version:


```r
install.packages("torch")
```

Does it work? Here's a quick test:


```r
library(torch)
torch_tensor(1)
```

```
## torch_tensor
##  1
## [ CPUFloatType{1} ]
```

Now, while `torch` contains all the core functionality, there is a whole ecosystem built -- and being built -- around it.

Notably, `torchvision` is essential to image-processing tasks. In this example, we don't use it much -- overtly, that is. It's used more prominently behind the scenes. Let's get it:


```r
install.packages("torchvision")
```


```r
library(torchvision)
```

Finally, there is an evolving package, named `torchdatasets` , that wraps datasets in a convenient format, rendering them immediately usable from `torch`. Let's get this as well, as we're going to use one of the datasets it provides.


```r
remotes::install_github("mlverse/torchdatasets")
```


```r
library(torchdatasets)
```

# Get the dataset

"Guess the correlation" is a fun dataset that tasks one -- a person, if they feel like, or a program, if we train it -- to estimate the (linear) correlation between two variables displayed in a scatterplot.

`torchdatasets` will download, unpack, and preprocess it for us.

The training set is huge; it has 150000 observations. For instruction purposes, we don't really need so much data -- we'll restrict ourselves to small subsets, for each of training, validation, and test sets.


```r
train_indices <- 1:10000
val_indices <- 10001:15000
test_indices <- 15001:20000
```

Now, the following snippet does the following:

-   download and unpack the dataset,

-   do some custom preprocessing on the images (on top of what is already done by default) -- more on that soon

-   take just the first 10000 observations and put them in a `torch` `Dataset` object named `train_ds`.


```r
add_channel_dim <- function(img) img$unsqueeze(1)
crop_axes <- function(img) transform_crop(img, top = 0, left = 21, height = 131, width = 130)

root <- file.path(tempdir(), "correlation")

train_ds <- guess_the_correlation_dataset(
    # where to unpack
    root = root,
    # change to whereever you're keeping kaggle.json
    token = file.path(Sys.getenv("HOME"), ".kaggle/kaggle.json"),
    # additional preprocessing 
    transform = function(img) crop_axes(img) %>% add_channel_dim(),
    # don't take all data, but just the indices we pass in
    indexes = train_indices,
    download = TRUE
  )
```

As we're at it, let's do analogously for validation and test sets. We don't need to download again, as we're building on the same underlying data We just pick different observations.


```r
valid_ds <- guess_the_correlation_dataset(
    root = root,
    transform = function(img) crop_axes(img) %>% add_channel_dim(),
    indexes = val_indices,
    download = FALSE
  )

test_ds <- guess_the_correlation_dataset(
    root = root,
    transform = function(img) crop_axes(img) %>% add_channel_dim(),
    indexes = test_indices,
    download = FALSE
  )
```


```r
length(train_ds)
```

```
## [1] 10000
```


```r
train_ds[1]$x 
```

```
## torch_tensor
## (1,.,.) = 
##  Columns 1 to 9  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
##   0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
##   0.0000  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999  0.9999
## ... [the output was truncated (use n=-1 to disable)]
## [ CPUFloatType{1,130,130} ]
```


```r
train_ds[1]$y
```

```
## torch_tensor
## -0.45781
## [ CPUFloatType{} ]
```


```r
train_dl <- dataloader(train_ds, batch_size = 64, shuffle = TRUE)
length(train_dl)
```

```
## [1] 157
```

```r
iter <- train_dl$.iter()
batch <-iter$.next()
dim(batch$x)
```

```
## [1]  64   1 130 130
```


```r
par(mfrow = c(8,8), mar = rep(0, 4))
images <- as.array(batch$x$squeeze(2))
images %>%
  purrr::array_tree(1) %>%
  purrr::map(as.raster) %>%
  purrr::iwalk(~{plot(.x)})
```

<img src="index_files/figure-html/unnamed-chunk-14-1.png" width="672" />


```r
valid_dl <- dataloader(valid_ds, batch_size = 64)
length(valid_dl)
```

```
## [1] 79
```




```r
test_dl <- dataloader(test_ds, batch_size = 64)
length(test_dl)
```

```
## [1] 79
```


```r
net <- nn_module(
  
  "corr-cnn",
  
  initialize = function() {
    # in_channels, out_channels, kernel_size, stride = 1, padding = 0
    self$conv1 <- nn_conv2d(1, 32, 3)
    self$conv2 <- nn_conv2d(32, 64, 3)
    self$conv3 <- nn_conv2d(64, 64, 3)
    #self$dropout1 <- nn_dropout2d(0.25)
    #self$dropout2 <- nn_dropout2d(0.5)
    #self$conv4 <- nn_conv2d(64, 1, 1)
    self$fc1 <- nn_linear(14 * 14 * 64, 128)
    self$fc2 <- nn_linear(128, 1)
  },
  
  forward = function(x) {
    x %>% 
      self$conv1() %>%
      nnf_relu() %>%
      nnf_avg_pool2d(2) %>%
      self$conv2() %>%
      nnf_relu() %>%
      nnf_avg_pool2d(2) %>%
      self$conv3() %>%
      nnf_relu() %>%
      nnf_avg_pool2d(2) %>%
      #self$conv4() %>%
      nnf_relu() %>%
      #self$dropout1() %>%
      torch_flatten(start_dim = 2) %>%
      self$fc1() %>%
      nnf_relu() %>%
      #self$dropout2() %>%
      self$fc2()
  }
)
```


```r
model <- net()
```


```r
model(batch$x)
```

```
## torch_tensor
## 0.01 *
## -5.8107
##  -5.9084
##  -5.8387
##  -5.7937
##  -5.8549
##  -5.8586
##  -5.6906
##  -5.7825
##  -5.8360
##  -5.9043
##  -5.9816
##  -5.7571
##  -5.9465
##  -5.7325
##  -5.7074
##  -5.9316
##  -5.9719
##  -5.8181
##  -5.8158
##  -5.8182
##  -5.9384
##  -5.7190
##  -5.8665
##  -5.8611
##  -5.9330
##  -5.8023
##  -5.7801
##  -5.9413
##  -5.8153
## ... [the output was truncated (use n=-1 to disable)]
## [ CPUFloatType{64,1} ]
```


```r
optimizer <- optim_adam(model$parameters)
```


```r
i <- 1

train_batch <- function(b) {

  optimizer$zero_grad()
  output <- model(b$x)
  loss <- nnf_mse_loss(output, b$y$unsqueeze(2))
  
  if (i %% 10 == 0) cat(sprintf("\nBatch loss: batch: %d %1.5f\n", i, loss$item()))
  i <<- i + 1        
                       
  loss$backward()
  optimizer$step()
  loss$item()

}

valid_batch <- function(b) {

  output <- model(b$x)
  loss <- nnf_mse_loss(output, b$y$unsqueeze(2))
  loss$item()
}

num_epochs <- 1

for (epoch in 1:num_epochs) {

  model$train()
  train_losses <- c()

  for (b in enumerate(train_dl)) {
    loss <- train_batch(b)
    train_losses <- c(train_losses, loss)
  }

  model$eval()
  valid_losses <- c()

  for (b in enumerate(valid_dl)) {
    loss <- valid_batch(b)
    valid_losses <- c(valid_losses, loss)
  }

  cat(sprintf("\nLoss at epoch %d: training: %1.5f, validation: %1.5f\n", epoch, mean(train_losses), mean(valid_losses)))
}
```

```
## 
## Batch loss: batch: 10 0.19578
## 
## Batch loss: batch: 20 0.15609
## 
## Batch loss: batch: 30 0.02771
## 
## Batch loss: batch: 40 0.02623
## 
## Batch loss: batch: 50 0.02547
## 
## Batch loss: batch: 60 0.02206
## 
## Batch loss: batch: 70 0.01569
## 
## Batch loss: batch: 80 0.01972
## 
## Batch loss: batch: 90 0.01847
## 
## Batch loss: batch: 100 0.01178
## 
## Batch loss: batch: 110 0.01803
## 
## Batch loss: batch: 120 0.01124
## 
## Batch loss: batch: 130 0.01066
## 
## Batch loss: batch: 140 0.01053
## 
## Batch loss: batch: 150 0.01149
## 
## Loss at epoch 1: training: 0.04701, validation: 0.01090
```


```r
model$eval()

i <- 1

test_batch <- function(b) {

  output <- model(b$x)
  loss <- nnf_mse_loss(output, b$y$unsqueeze(2))
  
  test_losses <<- c(test_losses, loss$item())
  
  if (i %% 20 == 0) cat(sprintf("\nBatch loss: batch: %d %1.5f\n", i, loss$item()))
  i <<- i + 1        
  
}

test_losses <- c()

for (b in enumerate(test_dl)) {
  test_batch(b)
}
```

```
## 
## Batch loss: batch: 20 0.01083
## 
## Batch loss: batch: 40 0.01389
## 
## Batch loss: batch: 60 0.01069
```

```r
mean(test_losses)
```

```
## [1] 0.01115602
```
