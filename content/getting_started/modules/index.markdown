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
## -0.1325  0.1399  0.4886
## [ CPUFloatType{1,3} ]
## 
## $bias
## torch_tensor
##  0.4913
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
##  0.7682
## -0.3203
##  1.0580
##  0.8967
##  0.5858
##  0.9522
##  1.6417
##  0.5137
##  1.1851
##  0.0160
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
##  43.2585  99.9657  31.9080
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
## -0.0110  0.2332  0.2889
## -0.1459  0.4978  0.2462
##  0.1636  0.1208  0.0649
##  0.1926 -0.4768 -0.2680
##  0.0684 -0.0201  0.4688
## -0.1161  0.2074  0.4727
## -0.0659  0.1491 -0.5230
##  0.2531 -0.4527  0.3616
## -0.2247 -0.1439  0.1411
##  0.5493 -0.0718 -0.4383
## -0.3387 -0.1974 -0.5760
##  0.5070 -0.4525  0.3350
##  0.2855  0.3381 -0.5304
## -0.1988 -0.0483  0.2689
## -0.0023 -0.5428 -0.1017
## -0.1302  0.2584  0.2455
## [ CPUFloatType{16,3} ]
## 
## $`0.bias`
## torch_tensor
## -0.0193
##  0.5503
## -0.5541
## -0.0698
##  0.0738
##  0.0443
##  0.3107
## -0.0098
## -0.2106
##  0.4059
##  0.4992
##  0.0689
## -0.2694
## -0.0895
## -0.2234
##  0.4339
## [ CPUFloatType{16} ]
## 
## $`2.weight`
## torch_tensor
## Columns 1 to 10 0.2087  0.0468  0.1054  0.0627 -0.2169  0.0206  0.2412  0.2036 -0.1649 -0.1189
## 
## Columns 11 to 16 0.2433  0.0918  0.0720  0.2166  0.1431 -0.0540
## [ CPUFloatType{1,16} ]
## 
## $`2.bias`
## torch_tensor
##  0.2180
## [ CPUFloatType{1} ]
```

To inspect an individual parameter, make use of its position in the
sequential model. For example:


```r
model[[1]]$bias
```

```
## torch_tensor
## -0.0193
##  0.5503
## -0.5541
## -0.0698
##  0.0738
##  0.0443
##  0.3107
## -0.0098
## -0.2106
##  0.4059
##  0.4992
##  0.0689
## -0.2694
## -0.0895
## -0.2234
##  0.4339
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
##  16.6937
##   4.6757
##   0.0000
##   0.0000
## -15.1822
##   1.6513
##  14.4723
##   6.1088
##   0.0000
##  -8.3257
##   9.7306
##   2.7535
##   2.8815
##  10.8292
##   0.0000
##  -5.4011
## [ CPUFloatType{16} ]
```

And placing the composite module on the GPU will move all tensors there:


```r
model$cuda()
model[[1]]$bias$grad
```

```
## torch_tensor
##  16.6937
##   4.6757
##   0.0000
##   0.0000
## -15.1822
##   1.6513
##  14.4723
##   6.1088
##   0.0000
##  -8.3257
##   9.7306
##   2.7535
##   2.8815
##  10.8292
##   0.0000
##  -5.4011
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
## Epoch:  10    Loss:  222.374 
## Epoch:  20    Loss:  163.3627 
## Epoch:  30    Loss:  128.4028 
## Epoch:  40    Loss:  107.4459 
## Epoch:  50    Loss:  95.65485 
## Epoch:  60    Loss:  89.41197 
## Epoch:  70    Loss:  86.229 
## Epoch:  80    Loss:  84.62106 
## Epoch:  90    Loss:  83.77222 
## Epoch:  100    Loss:  83.27081 
## Epoch:  110    Loss:  82.93011 
## Epoch:  120    Loss:  82.66684 
## Epoch:  130    Loss:  82.4418 
## Epoch:  140    Loss:  82.23875 
## Epoch:  150    Loss:  82.05148 
## Epoch:  160    Loss:  81.87491 
## Epoch:  170    Loss:  81.70572 
## Epoch:  180    Loss:  81.54685 
## Epoch:  190    Loss:  81.40156 
## Epoch:  200    Loss:  81.26586
```

The forward pass looks a lot better now; however, we still loop through
the model's parameters and update each one by hand. Furthermore, you may
be already be suspecting that `torch` provides abstractions for common
loss functions. In the final tutorial, we'll address both points, making
use of `torch` losses and optimizers.
