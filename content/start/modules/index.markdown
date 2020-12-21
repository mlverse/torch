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

## Modules

From other frameworks (Keras, say), you may be used to distinguishing
between *models* and *layers*. In `torch`, both are instances of
`nn_Module()`, and thus, have some methods in common. For those thinking
in terms of "models" and "layers", I'm artificially splitting up this
section into two parts. In reality though, there is no dichotomy: New
modules may be composed of existing ones up to arbitrary levels of
recursion.

#### Base modules ("layers")

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
## -0.4244  0.1499  0.4805
## [ CPUFloatType{1,3} ]
## 
## $bias
## torch_tensor
## 0.01 *
##  1.6601
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
##  1.7770
##  0.6039
##  0.7412
## -0.3740
##  1.5227
## -0.1166
##  0.3251
## -0.6749
## -0.1598
##  0.8503
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
##   3.7412  37.1724  81.7988
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

#### Container modules ("models")

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
##  0.0796 -0.1433  0.4188
##  0.1805 -0.2711  0.2769
## -0.1158  0.5639 -0.3620
## -0.4147 -0.0566 -0.4255
##  0.2575  0.2169  0.4878
##  0.4275  0.5577 -0.2251
## -0.0853  0.3706  0.3121
## -0.0927  0.2561 -0.2062
##  0.1545 -0.5552 -0.2774
##  0.3296 -0.4009  0.2086
##  0.4065  0.3580  0.2244
## -0.0229  0.0824  0.0176
## -0.0046  0.0795  0.2848
## -0.3591 -0.5654 -0.1917
## -0.2336 -0.4256 -0.5334
## -0.1276  0.4190 -0.4651
## [ CPUFloatType{16,3} ]
## 
## $`0.bias`
## torch_tensor
## -0.2169
## -0.1783
##  0.3893
## -0.3595
##  0.2069
##  0.4184
## -0.4101
## -0.0554
##  0.3975
## -0.0123
## -0.1127
## -0.1022
##  0.1546
##  0.0893
## -0.2340
##  0.5517
## [ CPUFloatType{16} ]
## 
## $`2.weight`
## torch_tensor
## Columns 1 to 10-0.1046 -0.0773  0.1901  0.1578 -0.0164  0.2003  0.0496  0.1752 -0.1802 -0.1733
## 
## Columns 11 to 16-0.2302 -0.0761 -0.0676 -0.0405  0.2189  0.1496
## [ CPUFloatType{1,16} ]
## 
## $`2.bias`
## torch_tensor
## 0.01 *
##  6.8527
## [ CPUFloatType{1} ]
```

To inspect an individual parameter, make use of its position in the
sequential model. For example:


```r
model[[1]]$bias
```

```
## torch_tensor
## -0.2169
## -0.1783
##  0.3893
## -0.3595
##  0.2069
##  0.4184
## -0.4101
## -0.0554
##  0.3975
## -0.0123
## -0.1127
## -0.1022
##  0.1546
##  0.0893
## -0.2340
##  0.5517
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
##  -6.2734
##  -3.0937
##  13.3040
##   0.0000
##  -1.6402
##  16.0200
##   1.4880
##   5.2569
## -10.8122
## -10.3979
## -13.8123
##  -1.5222
##  -6.7553
##  -1.6214
##   2.1892
##  11.9643
## [ CPUFloatType{16} ]
```

And placing the composite module on the GPU will move all tensors there:


```r
model$cuda()
model[[1]]$bias$grad
```

```
## torch_tensor
##  -6.2734
##  -3.0937
##  13.3040
##   0.0000
##  -1.6402
##  16.0200
##   1.4880
##   5.2569
## -10.8122
## -10.3979
## -13.8123
##  -1.5222
##  -6.7553
##  -1.6214
##   2.1892
##  11.9643
## [ CUDAFloatType{16} ]
```

Now let's see how using `nn_sequential()` can simplify our example
network.

## Simple network using modules


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
## Epoch:  10    Loss:  229.4971 
## Epoch:  20    Loss:  180.9762 
## Epoch:  30    Loss:  147.2483 
## Epoch:  40    Loss:  124.2533 
## Epoch:  50    Loss:  109.2876 
## Epoch:  60    Loss:  100.0198 
## Epoch:  70    Loss:  94.53094 
## Epoch:  80    Loss:  91.27599 
## Epoch:  90    Loss:  89.28571 
## Epoch:  100    Loss:  87.99605 
## Epoch:  110    Loss:  87.11208 
## Epoch:  120    Loss:  86.47411 
## Epoch:  130    Loss:  85.99896 
## Epoch:  140    Loss:  85.62224 
## Epoch:  150    Loss:  85.30872 
## Epoch:  160    Loss:  85.05437 
## Epoch:  170    Loss:  84.83028 
## Epoch:  180    Loss:  84.63102 
## Epoch:  190    Loss:  84.4514 
## Epoch:  200    Loss:  84.28427
```

The forward pass looks a lot better now; however, we still loop through
the model's parameters and update each one by hand. Furthermore, you may
be already be suspecting that `torch` provides abstractions for common
loss functions. In the final tutorial, we'll address both points, making
use of `torch` losses and optimizers.
