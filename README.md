
<!-- README.md is generated from README.Rmd. Please edit that file -->

torch <a href='https://torch.mlverse.org'><img src='man/figures/torch.png' align="right" height="139" /></a>
============================================================================================================

[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://www.tidyverse.org/lifecycle/#experimental)
![R build
status](https://github.com/mlverse/torch/workflows/Test/badge.svg)
[![CRAN
status](https://www.r-pkg.org/badges/version/torch)](https://CRAN.R-project.org/package=torch)
[![](https://cranlogs.r-pkg.org/badges/torch)](https://cran.r-project.org/package=torch)

Installation
------------

Run:

    remotes::install_github("mlverse/torch")

At the first package load additional software will be installed.

Example
-------

Currently this package is only a proof of concept and you can only
create a Torch Tensor from an R object. And then convert back from a
torch Tensor to an R object.

    library(torch)
    x <- array(runif(8), dim = c(2, 2, 2))
    y <- torch_tensor(x, dtype = torch_float64())
    y
    #> torch_tensor 
    #> (1,.,.) = 
    #>   0.5406  0.8648
    #>   0.3097  0.9715
    #> 
    #> (2,.,.) = 
    #>   0.1309  0.8992
    #>   0.4849  0.1902
    #> [ CPUDoubleType{2,2,2} ]
    identical(x, as_array(y))
    #> [1] TRUE

### Simple Autograd Example

In the following snippet we let torch, using the autograd feature,
calculate the derivatives:

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

### Linear Regression

In the following example we are going to fit a linear regression from
scratch using torch’s Autograd.

**Note** all methods that end with `_` (eg. `sub_`), will modify the
tensors in place.

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
    #> 1e-09 *
    #>  5.2672
    #>  -6.7969
    #> [ CPUFloatType{2,1} ]
    print(b) 
    #> torch_tensor 
    #> 0.01 *
    #> -9.6802
    #> [ CPUFloatType{1} ]

Contributing
------------

No matter your current skills it’s possible to contribute to `torch`
development. See the contributing guide for more information.
