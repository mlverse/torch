
<!-- README.md is generated from README.Rmd. Please edit that file -->

# torch

[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://www.tidyverse.org/lifecycle/#experimental)
![R build
status](https://github.com/mlverse/torch/workflows/Test/badge.svg)

## Installation

Run:

``` r
remotes::install_github("mlverse/torch")
```

At the first package load additional software will be installed.

## Example

Currently this package is only a prrof of concept and you can only
create a Torch Tensor from an R object. And then convert back from a
torch Tensor to an R object.

``` r
library(torchr)
x <- array(runif(8), dim = c(2, 2, 2))
y <- torch_tensor(x, dtype = torch_float64())
y
#> torch_tensor 
#> (1,.,.) = 
#>   0.5178  0.2795
#>   0.9975  0.9788
#> 
#> (2,.,.) = 
#>   0.9774  0.7181
#>   0.4617  0.7350
#> [ CPUDoubleType{2,2,2} ]
identical(x, as_array(y))
#> [1] TRUE
```

### Simple Autograd Example

In the following snippet we let torch, using the autograd feature,
calculate the derivatives:

``` r
x <- torch_tensor(1, requires_grad = TRUE)
w <- torch_tensor(2, requires_grad = TRUE)
b <- torch_tensor(3, requires_grad = TRUE)
y <- w * x + b
y$backward()
x$grad()
#> torch_tensor 
#>  2
#> [ CPUFloatType{1} ]
w$grad()
#> torch_tensor 
#>  1
#> [ CPUFloatType{1} ]
b$grad()
#> torch_tensor 
#>  1
#> [ CPUFloatType{1} ]
```

### Linear Regression

In the following example we are going to fit a linear regression from
scratch using torch’s Autograd.

**Note** all methods that end with `_` (eg. `sub_`), will modify the
tensors in place.

``` r
x <- matrix(runif(100), ncol = 2)
y <- matrix(0.1 + 0.5 * x[,1] - 0.7 * x[,2], ncol = 1)
x_t <- torch_tensor(x)
y_t <- torch_tensor(y)
w <- torch_tensor(matrix(rnorm(2), nrow = 2), requires_grad = TRUE)
b <- torch_tensor(0, requires_grad = TRUE)
lr <- 0.5
for (i in 1:100) {
  y_hat <- torch_mm(x_t, w) + b
  loss <- torch_mean((y_t - y_hat)^2)
  
  loss$backward()
  
  with_no_grad({
    w$sub_(w$grad()*lr)
    b$sub_(b$grad()*lr)   
  })
  
  w$grad()$zero_()
  b$grad()$zero_()
}
print(w)
#> torch_tensor 
#>  0.5011
#> -0.6997
#> [ CPUFloatType{2,1} ]
print(b) 
#> torch_tensor 
#> 0.01 *
#>  9.9441
#> [ CPUFloatType{1} ]
```
