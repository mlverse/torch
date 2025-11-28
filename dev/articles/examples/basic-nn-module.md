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
    ##  0.8826
    ##  0.4781
    ##  0.2548
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
    ##  0.8826
    ##  0.4781
    ##  0.2548
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
    ## -1.2811
    ## -0.4946
    ## -1.6332
    ## -2.0593
    ##  0.6708
    ##  1.0021
    ## -0.2444
    ##  0.7206
    ##  0.9381
    ## -0.1291
    ## [ CPUFloatType{10,1} ][ grad_fn = <AddBackward0> ]
