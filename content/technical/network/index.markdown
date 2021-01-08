---
title: "Neural network from scratch"
weight: 1
description: | 
  Code a neural network from scratch.
---

Our first version of the network will make use of "just" one torch feature: tensors. In fact, "just" doesn't quite fit as a qualifier, as tensors really are all we need! However, you'll soon see how much easier it gets through functionality built on top: automatic differentiation, optimizers, and neural network modules. For now though, just take this as a holistic example, as well as a motivation to learn more about tensors in the upcoming section!

To see tensors in action, we don't even have to wait until we code the network. Definitely, we need some data for it to work on, and these already we can simulate with torch.

# Generate random data

We use `torch_randn()` to simulate standard normally-distributed data, of a desired shape. For example:


```r
library(torch)
torch_randn(2, 3, 4)
```

```
## torch_tensor
## (1,.,.) = 
##   0.7548 -0.2971  2.3196 -0.4786
##   0.9237  1.2791  1.3958  1.0573
##   0.4980  0.0116  0.3937  0.2128
## 
## (2,.,.) = 
##   0.1766 -1.6569  0.4122 -0.2328
##  -0.1308 -0.1602  1.2324 -0.5125
##  -0.4426  0.0146  0.1818  0.2237
## [ CPUFloatType{2,3,4} ]
```

For our example, we want to have three input features ...


```r
# input dimensionality (number of input features)
d_in <- 3
```

... and we want to predict a single outcome.


```r
# output dimensionality (number of predicted features)
d_out <- 1
```

We will have a hundred observations in the training set.


```r
# number of observations in training set
n <- 100
```

So we create the data, with input `x` normally distributed and outcome `y`, dependent on all three features but a bit noisy:


```r
# create random data
# input
x <- torch_randn(n, d_in)
# target
y <- x[, 1, drop = FALSE] * 0.2 - x[, 2, drop = FALSE] * 1.3 - x[, 3, drop = FALSE] * 0.5 + torch_randn(n, 1)
```

# Initialize weights

`torch_randn()` is one of several functions used to initialize tensors of arbitrary shape. Seeing how we're at it, there is another place where we need to do something like this. With neural networks, it's all about the *weights*: those updateable parameters that determine how an intermediate result calculated by layer `n`'s units influences the units in layer `n+1`.

There are two types of weights. The first, the one we often restrict the term *weights* to, is different for each connection. So if we want a hidden layer with 32 units, we need a weight matrix of shape 100 (number of observations) by 32. That matrix will be updated during training, but we need to initialize it; and that, again, is accomplished using `torch_rand()`:


```r
# dimensionality of hidden layer
d_hidden <- 32

# weights connecting input to hidden layer
w1 <- torch_randn(d_in, d_hidden)
```

The hidden layer, in turn, is connected to an output layer of a single unit. The corresponding weight matrix, then, has size `32x1`:


```r
# weights connecting hidden to output layer
w2 <- torch_randn(d_hidden, d_out)
```

Now, the second type of weight, called *bias*, is not per connection, but per *unit*. Those biases we initialize to all zeroes:


```r
# hidden layer bias
b1 <- torch_zeros(1, d_hidden)
# output layer bias
b2 <- torch_zeros(1, d_out)
```

# Training loop

Here are the four phases of the training loop -- forward pass, determination of the loss, backward pass, and weight updates --, now with all operations being `torch` tensor methods. Firstly, the forward pass:


```r
  # compute pre-activations of hidden layers (dim: 100 x 32)
  # torch_mm does matrix multiplication
  h <- x$mm(w1) + b1
  
  # apply activation function (dim: 100 x 32)
  # torch_clamp cuts off values below/above given thresholds
  h_relu <- h$clamp(min = 0)
  
  # compute output (dim: 100 x 1)
  y_pred <- h_relu$mm(w2) + b2
```

Loss computation:


```r
  loss <- as.numeric((y_pred - y)$pow(2)$sum())
```

Backprop:


```r
  # gradient of loss w.r.t. prediction (dim: 100 x 1)
  grad_y_pred <- 2 * (y_pred - y)
  # gradient of loss w.r.t. w2 (dim: 32 x 1)
  grad_w2 <- h_relu$t()$mm(grad_y_pred)
  # gradient of loss w.r.t. hidden activation (dim: 100 x 32)
  grad_h_relu <- grad_y_pred$mm(w2$t())
  # gradient of loss w.r.t. hidden pre-activation (dim: 100 x 32)
  grad_h <- grad_h_relu$clone()
  
  grad_h[h < 0] <- 0
  
  # gradient of loss w.r.t. b2 (shape: ())
  grad_b2 <- grad_y_pred$sum()
  
  # gradient of loss w.r.t. w1 (dim: 3 x 32)
  grad_w1 <- x$t()$mm(grad_h)
  # gradient of loss w.r.t. b1 (shape: (32, ))
  grad_b1 <- grad_h$sum(dim = 1)
```

And weight updates:


```r
  learning_rate <- 1e-4
  
  w2 <- w2 - learning_rate * grad_w2
  b2 <- b2 - learning_rate * grad_b2
  w1 <- w1 - learning_rate * grad_w1
  b1 <- b1 - learning_rate * grad_b1
```

Finally, let's put the pieces together.

## Complete network using `torch` tensors


```r
library(torch)

### generate training data -----------------------------------------------------

# input dimensionality (number of input features)
d_in <- 3
# output dimensionality (number of predicted features)
d_out <- 1
# number of observations in training set
n <- 100


# create random data
x <- torch_randn(n, d_in)
y <-
  x[, 1, NULL] * 0.2 - x[, 2, NULL] * 1.3 - x[, 3, NULL] * 0.5 + torch_randn(n, 1)


### initialize weights ---------------------------------------------------------

# dimensionality of hidden layer
d_hidden <- 32
# weights connecting input to hidden layer
w1 <- torch_randn(d_in, d_hidden)
# weights connecting hidden to output layer
w2 <- torch_randn(d_hidden, d_out)

# hidden layer bias
b1 <- torch_zeros(1, d_hidden)
# output layer bias
b2 <- torch_zeros(1, d_out)

### network parameters ---------------------------------------------------------

learning_rate <- 1e-4

### training loop --------------------------------------------------------------

for (t in 1:200) {
  ### -------- Forward pass --------
  
  # compute pre-activations of hidden layers (dim: 100 x 32)
  h <- x$mm(w1) + b1
  # apply activation function (dim: 100 x 32)
  h_relu <- h$clamp(min = 0)
  # compute output (dim: 100 x 1)
  y_pred <- h_relu$mm(w2) + b2
  
  ### -------- compute loss --------

  loss <- as.numeric((y_pred - y)$pow(2)$sum())
  
  if (t %% 10 == 0)
    cat("Epoch: ", t, "   Loss: ", loss, "\n")
  
  ### -------- Backpropagation --------
  
  # gradient of loss w.r.t. prediction (dim: 100 x 1)
  grad_y_pred <- 2 * (y_pred - y)
  # gradient of loss w.r.t. w2 (dim: 32 x 1)
  grad_w2 <- h_relu$t()$mm(grad_y_pred)
  # gradient of loss w.r.t. hidden activation (dim: 100 x 32)
  grad_h_relu <- grad_y_pred$mm(
    w2$t())
  # gradient of loss w.r.t. hidden pre-activation (dim: 100 x 32)
  grad_h <- grad_h_relu$clone()
  
  grad_h[h < 0] <- 0
  
  # gradient of loss w.r.t. b2 (shape: ())
  grad_b2 <- grad_y_pred$sum()
  
  # gradient of loss w.r.t. w1 (dim: 3 x 32)
  grad_w1 <- x$t()$mm(grad_h)
  # gradient of loss w.r.t. b1 (shape: (32, ))
  grad_b1 <- grad_h$sum(dim = 1)
  
  ### -------- Update weights --------
  
  w2 <- w2 - learning_rate * grad_w2
  b2 <- b2 - learning_rate * grad_b2
  w1 <- w1 - learning_rate * grad_w1
  b1 <- b1 - learning_rate * grad_b1
  
}
```

```
## Epoch:  10    Loss:  283.2281 
## Epoch:  20    Loss:  184.7813 
## Epoch:  30    Loss:  148.6949 
## Epoch:  40    Loss:  132.085 
## Epoch:  50    Loss:  122.7631 
## Epoch:  60    Loss:  116.8454 
## Epoch:  70    Loss:  112.5663 
## Epoch:  80    Loss:  109.0198 
## Epoch:  90    Loss:  106.0655 
## Epoch:  100    Loss:  103.6448 
## Epoch:  110    Loss:  101.5424 
## Epoch:  120    Loss:  99.66808 
## Epoch:  130    Loss:  97.93629 
## Epoch:  140    Loss:  96.41391 
## Epoch:  150    Loss:  95.0442 
## Epoch:  160    Loss:  93.81817 
## Epoch:  170    Loss:  92.76872 
## Epoch:  180    Loss:  91.68913 
## Epoch:  190    Loss:  90.63338 
## Epoch:  200    Loss:  89.64046
```

In the next tutorial, we'll make an important change, freeing us from having to think in detail about the backward pass.
