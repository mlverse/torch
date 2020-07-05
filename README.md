
<!-- README.md is generated from README.Rmd. Please edit that file -->

# torch <a href='https://mlverse.github.io/torch'><img src='man/figures/torch.png' align="right" height="139" /></a>

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

Currently this package is only a proof of concept and you can only
create a Torch Tensor from an R object. And then convert back from a
torch Tensor to an R object.

``` r
library(torch)
x <- array(runif(8), dim = c(2, 2, 2))
y <- torch_tensor(x, dtype = torch_float64())
y
#> torch_tensor 
#> (1,.,.) = 
#>   0.8687  0.0157
#>   0.4237  0.8971
#> 
#> (2,.,.) = 
#>   0.4021  0.5509
#>   0.3374  0.9034
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
x$grad
#> torch_tensor 
#>  2
#> [ CPUFloatType{1} ]
w$grad
#> torch_tensor 
#>  1
#> [ CPUFloatType{1} ]
b$grad
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
x <- torch_randn(100, 2)
y <- 0.1 + 0.5*x[,1] - 0.7*x[,2]

w <- torch_randn(2, 1, requires_grad = TRUE)
b <- torch_zeros(1, requires_grad = TRUE)

lr <- 0.5
for (i in 1:100) {
  y_hat <- torch_mm(x, w) + b
  loss <- torch_mean((y - y_hat$squeeze(1))^2)
  
  loss$backward()
  
  with_no_grad({
    w$sub_(w$grad*lr)
    b$sub_(b$grad*lr)   
    
    w$grad$zero_()
    b$grad$zero_()
  })
}
print(w)
#> torch_tensor 
#>  0.5000
#> -0.7000
#> [ CPUFloatType{2,1} ]
print(b) 
#> torch_tensor 
#> 0.01 *
#> 10.0000
#> [ CPUFloatType{1} ]
```

## Contributing

No matter your current skills it’s possible to contribute to `torch`
development. See the contributing guide for more information.
