---
title: "Modules"
weight: 4
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
##  0.2746  0.1806  0.5658
## [ CPUFloatType{1,3} ]
## 
## $bias
## torch_tensor
##  0.1348
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
##  0.0457
## -0.2944
##  0.1046
##  0.8796
## -0.3529
## -0.2735
## -0.3491
## -1.2709
## -0.2419
##  0.2618
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
## -64.5866  -4.5878 -17.3627
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
##  0.5432 -0.1786  0.2262
## -0.4251 -0.2114 -0.3005
##  0.5404  0.5167  0.4031
## -0.2928  0.2713 -0.5621
## -0.1538  0.3970  0.3493
## -0.4646  0.3377  0.0653
##  0.3810 -0.0525  0.0309
## -0.0062 -0.2525 -0.2091
## -0.0372 -0.2401 -0.0344
## -0.0682  0.0348 -0.3300
##  0.5520  0.3174 -0.5156
## -0.0116  0.1722  0.3682
## -0.1925 -0.3215  0.3936
## -0.1558 -0.3241  0.1210
##  0.0903 -0.3383  0.3782
##  0.2078 -0.3504  0.2801
## [ CPUFloatType{16,3} ]
## 
## $`0.bias`
## torch_tensor
## -0.2698
## -0.1447
## -0.2124
## -0.2239
##  0.1827
## -0.1008
## -0.0195
## -0.3044
##  0.1232
## -0.3131
## -0.2504
##  0.3246
## -0.2733
## -0.5319
## -0.4463
## -0.5719
## [ CPUFloatType{16} ]
## 
## $`2.weight`
## torch_tensor
## Columns 1 to 10-0.0515 -0.1148 -0.0378 -0.0299 -0.1381  0.1993 -0.0235 -0.0346 -0.0807 -0.2410
## 
## Columns 11 to 16-0.1232 -0.1432  0.2147  0.2096  0.0261  0.2043
## [ CPUFloatType{1,16} ]
## 
## $`2.bias`
## torch_tensor
## -0.2238
## [ CPUFloatType{1} ]
```

To inspect an individual parameter, make use of its position in the
sequential model. For example:


```r
model[[1]]$bias
```

```
## torch_tensor
## -0.2698
## -0.1447
## -0.2124
## -0.2239
##  0.1827
## -0.1008
## -0.0195
## -0.3044
##  0.1232
## -0.3131
## -0.2504
##  0.3246
## -0.2733
## -0.5319
## -0.4463
## -0.5719
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
##  -1.0297
##  -9.1879
##  -0.3781
##  -1.1946
##  -9.6673
##  11.9571
##  -0.4693
##  -0.6913
##  -5.6459
##  -2.4103
##  -3.6958
## -11.4523
##   8.5864
##   4.1915
##   0.0000
##   0.0000
## [ CPUFloatType{16} ]
```

And placing the composite module on the GPU will move all tensors there:


```r
model$cuda()
model[[1]]$bias$grad
```

```
## torch_tensor
##  -1.0297
##  -9.1879
##  -0.3781
##  -1.1946
##  -9.6673
##  11.9571
##  -0.4693
##  -0.6913
##  -5.6459
##  -2.4103
##  -3.6958
## -11.4523
##   8.5864
##   4.1915
##   0.0000
##   0.0000
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
## Epoch:  10    Loss:  180.9211 
## Epoch:  20    Loss:  142.5852 
## Epoch:  30    Loss:  118.842 
## Epoch:  40    Loss:  104.6096 
## Epoch:  50    Loss:  96.36818 
## Epoch:  60    Loss:  91.74362 
## Epoch:  70    Loss:  89.16847 
## Epoch:  80    Loss:  87.71291 
## Epoch:  90    Loss:  86.8581 
## Epoch:  100    Loss:  86.31103 
## Epoch:  110    Loss:  85.92104 
## Epoch:  120    Loss:  85.61875 
## Epoch:  130    Loss:  85.36913 
## Epoch:  140    Loss:  85.14122 
## Epoch:  150    Loss:  84.93426 
## Epoch:  160    Loss:  84.73933 
## Epoch:  170    Loss:  84.55376 
## Epoch:  180    Loss:  84.37543 
## Epoch:  190    Loss:  84.2056 
## Epoch:  200    Loss:  84.03828
```

The forward pass looks a lot better now; however, we still loop through
the model's parameters and update each one by hand. Furthermore, you may
be already be suspecting that `torch` provides abstractions for common
loss functions. In the final tutorial, we'll address both points, making
use of `torch` losses and optimizers.
