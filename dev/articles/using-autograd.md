# Using autograd

``` r
library(torch)
```

So far, all we’ve been using from torch is *tensors*, but we’ve been
performing all calculations ourselves – the computing the predictions,
the loss, the gradients (and thus, the necessary updates to the
weights), and the new weight values. In this chapter, we’ll make a
significant change: Namely, we spare ourselves the cumbersome
calculation of gradients, and have torch do it for us.

Before we see that in action, let’s get some more background.

## Automatic differentiation with autograd

Torch uses a module called *autograd* to record operations performed on
tensors, and store what has to be done to obtain the respective
gradients. These actions are stored as functions, and those functions
are applied in order when the gradient of the output (normally, the
loss) with respect to those tensors is calculated: starting from the
output node and *propagating* gradients *back* through the network. This
is a form of *reverse mode automatic differentiation*.

As users, we can see a bit of this implementation. As a prerequisite for
this “recording” to happen, tensors have to be created with
`requires_grad = TRUE`. E.g.

``` r
x <- torch_ones(2,2, requires_grad = TRUE)
```

To be clear, this is a tensor *with respect to which* gradients have to
be calculated – normally, a tensor representing a weight or a bias, not
the input data [¹](#fn1). If we now perform some operation on that
tensor, assigning the result to `y`

``` r
y <- x$mean()
```

we find that `y` now has a non-empty `grad_fn` that tells torch how to
compute the gradient of `y` with respect to `x`:

``` r
y$grad_fn
#> MeanBackward0
```

Actual computation of gradients is triggered by calling `backward()` on
the output tensor.

``` r
y$backward()
```

That executed, `x` now has a non-empty field `grad` that stores the
gradient of `y` with respect to `x`:

``` r
x$grad
#> torch_tensor
#>  0.2500  0.2500
#>  0.2500  0.2500
#> [ CPUFloatType{2,2} ]
```

With a longer chain of computations, we can peek at how torch builds up
a graph of backward operations.

Here is a slightly more complex example. We call `retain_grad()` on `y`
and `z` just for demonstration purposes; by default, intermediate
gradients – while of course they have to be computed – aren’t stored, in
order to save memory.

``` r
x1 <- torch_ones(2,2, requires_grad = TRUE)
x2 <- torch_tensor(1.1, requires_grad = TRUE)
y <- x1 * (x2 + 2)
y$retain_grad()
z <- y$pow(2) * 3
z$retain_grad()
out <- z$mean()
```

Starting from `out$grad_fn`, we can follow the graph all back to the
leaf nodes:

``` r
# how to compute the gradient for mean, the last operation executed
out$grad_fn
#> MeanBackward0
# how to compute the gradient for the multiplication by 3 in z = y$pow(2) * 3
out$grad_fn$next_functions
#> [[1]]
#> MulBackward1
# how to compute the gradient for pow in z = y.pow(2) * 3
out$grad_fn$next_functions[[1]]$next_functions
#> [[1]]
#> PowBackward0
# how to compute the gradient for the multiplication in y = x * (x + 2)
out$grad_fn$next_functions[[1]]$next_functions[[1]]$next_functions
#> [[1]]
#> MulBackward0
# how to compute the gradient for the two branches of y = x * (x + 2),
# where the left branch is a leaf node (AccumulateGrad for x1)
out$grad_fn$next_functions[[1]]$next_functions[[1]]$next_functions[[1]]$next_functions
#> [[1]]
#> torch::autograd::AccumulateGrad
#> 
#> [[2]]
#> AddBackward1
# here we arrive at the other leaf node (AccumulateGrad for x2)
out$grad_fn$next_functions[[1]]$next_functions[[1]]$next_functions[[1]]$next_functions[[2]]$next_functions
#> [[1]]
#> torch::autograd::AccumulateGrad
```

After calling `out$backward()`, all tensors in the graph will have their
respective gradients created. Without our calls to `retain_grad` above,
`z$grad` and `y$grad` would be empty:

``` r
out$backward()
z$grad
#> torch_tensor
#>  0.2500  0.2500
#>  0.2500  0.2500
#> [ CPUFloatType{2,2} ]
y$grad
#> torch_tensor
#>  4.6500  4.6500
#>  4.6500  4.6500
#> [ CPUFloatType{2,2} ]
x2$grad
#> torch_tensor
#>  18.6000
#> [ CPUFloatType{1} ]
x1$grad
#> torch_tensor
#>  14.4150  14.4150
#>  14.4150  14.4150
#> [ CPUFloatType{2,2} ]
```

Thus acquainted with autograd, we’re ready to modify our example.

## The simple network, now using autograd

For a single new line calling `loss$backward()`, now a number of lines
(that did manual backprop) are gone:

``` r
### generate training data -----------------------------------------------------
# input dimensionality (number of input features)
d_in <- 3
# output dimensionality (number of predicted features)
d_out <- 1
# number of observations in training set
n <- 100
# create random data
x <- torch_randn(n, d_in)
y <- x[,1]*0.2 - x[..,2]*1.3 - x[..,3]*0.5 + torch_randn(n)
y <- y$unsqueeze(dim = 1)
### initialize weights ---------------------------------------------------------
# dimensionality of hidden layer
d_hidden <- 32
# weights connecting input to hidden layer
w1 <- torch_randn(d_in, d_hidden, requires_grad = TRUE)
# weights connecting hidden to output layer
w2 <- torch_randn(d_hidden, d_out, requires_grad = TRUE)
# hidden layer bias
b1 <- torch_zeros(1, d_hidden, requires_grad = TRUE)
# output layer bias
b2 <- torch_zeros(1, d_out,requires_grad = TRUE)
### network parameters ---------------------------------------------------------
learning_rate <- 1e-4
### training loop --------------------------------------------------------------
for (t in 1:200) {

    ### -------- Forward pass -------- 
    y_pred <- x$mm(w1)$add(b1)$clamp(min = 0)$mm(w2)$add(b2)
    ### -------- compute loss -------- 
    loss <- (y_pred - y)$pow(2)$mean()
    if (t %% 10 == 0) cat(t, as_array(loss), "\n")
    ### -------- Backpropagation -------- 
    # compute the gradient of loss with respect to all tensors with requires_grad = True.
    loss$backward()
 
    ### -------- Update weights -------- 
    
    # Wrap in torch.no_grad() because this is a part we DON'T want to record for automatic gradient computation
    with_no_grad({
      
      w1$sub_(learning_rate * w1$grad)
      w2$sub_(learning_rate * w2$grad)
      b1$sub_(learning_rate * b1$grad)
      b2$sub_(learning_rate * b2$grad)
      
      # Zero the gradients after every pass, because they'd accumulate otherwise
      w1$grad$zero_()
      w2$grad$zero_()
      b1$grad$zero_()
      b2$grad$zero_()
    
    })
    
}
#> 10 23.60086 
#> 20 22.02482 
#> 30 20.60181 
#> 40 19.31541 
#> 50 18.15105 
#> 60 17.09586 
#> 70 16.13857 
#> 80 15.26791 
#> 90 14.47873 
#> 100 13.75948 
#> 110 13.10263 
#> 120 12.50075 
#> 130 11.94914 
#> 140 11.44309 
#> 150 10.97806 
#> 160 10.55 
#> 170 10.15615 
#> 180 9.792877 
#> 190 9.456893 
#> 200 9.145655
```

We still manually compute the forward pass, and we still manually update
the weights. In the last two chapters of this section, we’ll see how
these parts of the logic can be made more modular and reusable, as well.

------------------------------------------------------------------------

1.  Unless we *want* to change the data, as in adversarial example
    generation
