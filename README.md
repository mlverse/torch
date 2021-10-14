
<!-- README.md is generated from README.Rmd. Please edit that file -->

# torch <a href='https://torch.mlverse.org'><img src='man/figures/torch.png' align="right" height="139" /></a>

[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html)
![R build
status](https://github.com/mlverse/torch/workflows/Test/badge.svg)
[![CRAN
status](https://www.r-pkg.org/badges/version/torch)](https://CRAN.R-project.org/package=torch)
[![](https://cranlogs.r-pkg.org/badges/torch)](https://cran.r-project.org/package=torch)
[![Discord](https://img.shields.io/discord/837019024499277855?logo=discord)](https://discord.com/invite/s3D5cKhBkx)

## Installation

torch can be installed from CRAN with:

``` r
install.packages("torch")
```

You can also install the development version with:

``` r
remotes::install_github("mlverse/torch")
```

At the first package load additional software will be installed.

## Installation with Docker

If you would like to install with Docker, please read following
document.

-   [The way of installation with
    Docker](https://github.com/mlverse/torch/blob/master/docker/build_env_guide.md)

## Examples

You can create torch tensors from R objects with the `torch_tensor`
function and convert them back to R objects with `as_array`.

``` r
library(torch)
x <- array(runif(8), dim = c(2, 2, 2))
y <- torch_tensor(x, dtype = torch_float64())
y
#> torch_tensor
#> (1,.,.) = 
#>   0.5955  0.3436
#>   0.4946  0.4344
#> 
#> (2,.,.) = 
#>   0.9322  0.7824
#>   0.6503  0.7516
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

## Contributing

No matter your current skills it’s possible to contribute to `torch`
development. See the [contributing
guide](https://torch.mlverse.org/docs/contributing) for more
information.