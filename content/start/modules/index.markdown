---
title: "Using torch modules"
weight: 3
description: | 
  Create neural network modules.
---

In the first tutorial, we started learning about `torch` basics by coding a simple
neural network from scratch, making use of just a single of `torch`'s
features: *tensors*. Then, we immensely simplified the task, replacing
manual backpropagation with *autograd*. In this section, we *modularize*
the network - in both the habitual and a very literal sense: Low-level
matrix operations are swapped out for `torch` `module`s.

# Modules

From other frameworks (Keras, say), you may be used to distinguishing
between *models* and *layers*. In `torch`, both are instances of
`nn_Module()`, and thus, have some methods in common. For those thinking
in terms of "models" and "layers", I'm artificially splitting up this
section into two parts. In reality though, there is no dichotomy: New
modules may be composed of existing ones up to arbitrary levels of
recursion.

## Base modules ("layers")

Instead of writing out an affine operation by hand -- `x$mm(w1) + b1`,
say --, as we've been doing so far, we can create a linear module. The
following snippet instantiates a linear layer that expects three-feature
inputs and returns a single output per observation:


```r
library(torch)
l <- nn_linear(3, 1)
```

The module has two parameters, "weight" and "bias". Both now come
pre-initialized:


```r
l$parameters
```

```
## $weight
## torch_tensor
## -0.1019 -0.0990  0.1464
## [ CPUFloatType{1,3} ]
## 
## $bias
## torch_tensor
##  0.3637
## [ CPUFloatType{1} ]
```

Modules are callable; calling a module executes its `forward()` method,
which, for a linear layer, matrix-multiplies input and weights, and adds
the bias.

Let's try this:


```r
data  <- torch_randn(10, 3)
out <- l(data)
```

Unsurprisingly, `out` now holds some data:


```r
out$data()
```

```
## torch_tensor
## -0.0244
##  0.3721
##  0.5740
##  0.3929
##  0.4518
##  0.7229
##  0.4145
##  0.5695
##  0.4755
##  0.2048
## [ CPUFloatType{10,1} ]
```

In addition though, this tensor knows what will need to be done, should
ever it be asked to calculate gradients:


```r
out$grad_fn
```

```
## AddmmBackward
```

Note the difference between tensors returned by modules and self-created
ones. When creating tensors ourselves, we need to pass
`requires_grad = TRUE` to trigger gradient calculation. With modules,
`torch` correctly assumes that we'll want to perform backpropagation at
some point.

By now though, we haven't called `backward()` yet. Thus, no gradients
have yet been computed:


```r
l$weight$grad
```

```
## torch_tensor
## [ Tensor (undefined) ]
```

```r
l$bias$grad
```

```
## torch_tensor
## [ Tensor (undefined) ]
```

Let's change this:


```r
# out$backward() # error
```

Why the error? *Autograd* expects the output tensor to be a scalar,
while in our example, we have a tensor of size `(10, 1)`. This error
won't often occur in practice, where we work with *batches* of inputs
(sometimes, just a single batch). But still, it's interesting to see how
to resolve this.

To make the example work, we introduce a -- virtual -- final aggregation
step -- taking the mean, say. Let's call it `avg`. If such a mean were
taken, its gradient with respect to `l$weight` would be obtained via the
chain rule:

$$
\begin{equation*} 
 \frac{\partial \ avg}{\partial w} = \frac{\partial \ avg}{\partial \ out}  \ \frac{\partial \ out}{\partial w}
\end{equation*}`
$$

Of the quantities on the right side, we're interested in the second. We
need to provide the first one, the way it would look *if really we were
taking the mean*:


```r
d_avg_d_out <- torch_tensor(10)$`repeat`(10)$unsqueeze(1)$t()
out$backward(gradient = d_avg_d_out)
```

Now, `l$weight$grad` and `l$bias$grad` *do* contain gradients:


```r
l$weight$grad
```

```
## torch_tensor
## -1.7888 -55.8351 -3.7148
## [ CPUFloatType{1,3} ]
```

```r
l$bias$grad
```

```
## torch_tensor
##  100
## [ CPUFloatType{1} ]
```

In addition to `nn_linear()` , `torch` provides pretty much all the
common layers you might hope for. But few tasks are solved by a single
layer. How do you combine them? Or, in the usual lingo: How do you build
*models*?

## Container modules ("models")

Now, *models* are just modules that contain other modules. For example,
if all inputs are supposed to flow through the same nodes and along the
same edges, then `nn_sequential()` can be used to build a simple graph.

For example:


```r
model <- nn_sequential(
    nn_linear(3, 16),
    nn_relu(),
    nn_linear(16, 1)
)
```

We can use the same technique as above to get an overview of all model
parameters (two weight matrices and two bias vectors):


```r
model$parameters
```

```
## $`0.weight`
## torch_tensor
## -0.0362 -0.5047  0.0259
## -0.4210 -0.3794 -0.0850
## -0.5299 -0.0852  0.2957
##  0.1644  0.1044  0.3209
##  0.0185 -0.1925 -0.1361
## -0.1010  0.2942 -0.2920
##  0.2254  0.1517  0.4821
## -0.3828 -0.4848 -0.0008
##  0.2328 -0.5295  0.0869
##  0.0426  0.5333  0.2192
## -0.5353 -0.5647  0.5426
## -0.5480 -0.5726  0.4348
##  0.5555 -0.0041 -0.2384
##  0.1794  0.5171  0.3888
## -0.0657 -0.1766  0.1058
##  0.0159  0.4440  0.3025
## [ CPUFloatType{16,3} ]
## 
## $`0.bias`
## torch_tensor
## -0.3873
## -0.3990
##  0.5748
##  0.1864
## -0.3110
## -0.3023
## -0.3704
##  0.2794
##  0.5069
## -0.5375
##  0.1427
##  0.5329
##  0.4056
##  0.1006
##  0.1963
##  0.2450
## [ CPUFloatType{16} ]
## 
## $`2.weight`
## torch_tensor
## Columns 1 to 10-0.0345 -0.1588  0.0619 -0.0172 -0.1766  0.0332 -0.0634 -0.0022  0.0906  0.0684
## 
## Columns 11 to 16 0.1383  0.1413  0.2446 -0.0516 -0.2007 -0.1296
## [ CPUFloatType{1,16} ]
## 
## $`2.bias`
## torch_tensor
## 0.01 *
## -2.7460
## [ CPUFloatType{1} ]
```

To inspect an individual parameter, make use of its position in the
sequential model. For example:


```r
model[[1]]$bias
```

```
## torch_tensor
## -0.3873
## -0.3990
##  0.5748
##  0.1864
## -0.3110
## -0.3023
## -0.3704
##  0.2794
##  0.5069
## -0.5375
##  0.1427
##  0.5329
##  0.4056
##  0.1006
##  0.1963
##  0.2450
## [ CPUFloatType{16} ]
```

And just like `nn_linear()` above, this module can be called directly on
data:


```r
out <- model(data)
```

On a composite module like this one, calling `backward()` will
backpropagate through all the layers:


```r
out$backward(gradient = torch_tensor(10)$`repeat`(10)$unsqueeze(1)$t())

# e.g.
model[[1]]$bias$grad
```

```
## torch_tensor
##  -1.7237
##  -6.3515
##   5.5712
##  -1.0348
##  -5.2977
##   0.6642
##  -1.9010
##  -0.1569
##   7.2498
##   1.3682
##   9.6816
##  12.7196
##  17.1230
##  -2.0634
## -18.0636
##  -5.1826
## [ CPUFloatType{16} ]
```

And placing the composite module on the GPU will move all tensors there:


```r
model$cuda()
model[[1]]$bias$grad
```

```
## torch_tensor
##  -1.7237
##  -6.3515
##   5.5712
##  -1.0348
##  -5.2977
##   0.6642
##  -1.9010
##  -0.1569
##   7.2498
##   1.3682
##   9.6816
##  12.7196
##  17.1230
##  -2.0634
## -18.0636
##  -5.1826
## [ CUDAFloatType{16} ]
```

Now let's see how using `nn_sequential()` can simplify our example
network.

# Simple network using modules


```r
### generate training data -----------------------------------------------------

# input dimensionality (number of input features)
d_in <- 3
# output dimensionality (number of predicted features)
d_out <- 1
# number of observations in training set
n <- 100


# create random data
x <- torch_randn(n, d_in)
y <- x[, 1, NULL] * 0.2 - x[, 2, NULL] * 1.3 - x[, 3, NULL] * 0.5 + torch_randn(n, 1)


### define the network ---------------------------------------------------------

# dimensionality of hidden layer
d_hidden <- 32

model <- nn_sequential(
  nn_linear(d_in, d_hidden),
  nn_relu(),
  nn_linear(d_hidden, d_out)
)

### network parameters ---------------------------------------------------------

learning_rate <- 1e-4

### training loop --------------------------------------------------------------

for (t in 1:200) {
  
  ### -------- Forward pass -------- 
  
  y_pred <- model(x)
  
  ### -------- compute loss -------- 
  loss <- (y_pred - y)$pow(2)$sum()
  if (t %% 10 == 0)
    cat("Epoch: ", t, "   Loss: ", loss$item(), "\n")
  
  ### -------- Backpropagation -------- 
  
  # Zero the gradients before running the backward pass.
  model$zero_grad()
  
  # compute gradient of the loss w.r.t. all learnable parameters of the model
  loss$backward()
  
  ### -------- Update weights -------- 
  
  # Wrap in with_no_grad() because this is a part we DON'T want to record
  # for automatic gradient computation
  # Update each parameter by its `grad`
  
  with_no_grad({
    model$parameters %>% purrr::walk(function(param) param$sub_(learning_rate * param$grad))
  })
  
}
```

```
## Epoch:  10    Loss:  281.9471 
## Epoch:  20    Loss:  202.0964 
## Epoch:  30    Loss:  153.0894 
## Epoch:  40    Loss:  123.8677 
## Epoch:  50    Loss:  107.3094 
## Epoch:  60    Loss:  98.17577 
## Epoch:  70    Loss:  93.17717 
## Epoch:  80    Loss:  90.32096 
## Epoch:  90    Loss:  88.56997 
## Epoch:  100    Loss:  87.34862 
## Epoch:  110    Loss:  86.46484 
## Epoch:  120    Loss:  85.77603 
## Epoch:  130    Loss:  85.2014 
## Epoch:  140    Loss:  84.701 
## Epoch:  150    Loss:  84.25502 
## Epoch:  160    Loss:  83.84596 
## Epoch:  170    Loss:  83.46891 
## Epoch:  180    Loss:  83.11785 
## Epoch:  190    Loss:  82.79836 
## Epoch:  200    Loss:  82.49139
```

The forward pass looks a lot better now; however, we still loop through
the model's parameters and update each one by hand. Furthermore, you may
be already be suspecting that `torch` provides abstractions for common
loss functions. In the final tutorial, we'll address both points, making
use of `torch` losses and optimizers.
