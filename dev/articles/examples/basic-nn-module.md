# basic-nn-module

``` r
library(torch)

# creates example tensors. x requires_grad = TRUE tells that
# we are going to take derivatives over it.
dense <- nn_module(
  clasname = "dense",
  # the initialize function tuns whenever we instantiate the model
  initialize = function(in_features, out_features) {

    # just for you to see when this function is called
    cat("Calling initialize!")

    # we use nn_parameter to indicate that those tensors are special
    # and should be treated as parameters by `nn_module`.
    self$w <- nn_parameter(torch_randn(in_features, out_features))
    self$b <- nn_parameter(torch_zeros(out_features))

  },
  # this function is called whenever we call our model on input.
  forward = function(x) {
    cat("Calling forward!")
    torch_mm(x, self$w) + self$b
  }
)

model <- dense(3, 1)
```

    ## Calling initialize!

``` r
# you can get all parameters
model$parameters
```

    ## $w
    ## torch_tensor
    ##  1.9167
    ##  1.1706
    ## -0.4988
    ## [ CPUFloatType{3,1} ][ requires_grad = TRUE ]
    ## 
    ## $b
    ## torch_tensor
    ##  0
    ## [ CPUFloatType{1} ][ requires_grad = TRUE ]

``` r
# or individually
model$w
```

    ## torch_tensor
    ##  1.9167
    ##  1.1706
    ## -0.4988
    ## [ CPUFloatType{3,1} ][ requires_grad = TRUE ]

``` r
model$b
```

    ## torch_tensor
    ##  0
    ## [ CPUFloatType{1} ][ requires_grad = TRUE ]

``` r
# create an input tensor
x <- torch_randn(10, 3)
y_pred <- model(x)
```

    ## Calling forward!

``` r
y_pred
```

    ## torch_tensor
    ## -1.3133
    ##  2.4344
    ##  1.2372
    ## -2.9592
    ##  1.1354
    ## -2.2606
    ##  0.7070
    ##  1.9451
    ## -3.4449
    ##  1.3424
    ## [ CPUFloatType{10,1} ][ grad_fn = <AddBackward0> ]
