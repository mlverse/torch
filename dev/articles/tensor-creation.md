# Creating tensors

``` r
library(torch)
```

In this article we describe various ways of creating `torch` tensors in
R.

## From R objects

You can create tensors from R objects using the `torch_tensor` function.
The `torch_tensor` function takes an R vector, matrix or array and
creates an equivalent `torch_tensor`.

You can see a few examples below:

``` r
torch_tensor(c(1,2,3))
#> torch_tensor
#>  1
#>  2
#>  3
#> [ CPUFloatType{3} ]

# conform to row-major indexing used in torch
torch_tensor(matrix(1:10, ncol = 5, nrow = 2, byrow = TRUE))
#> torch_tensor
#>   1   2   3   4   5
#>   6   7   8   9  10
#> [ CPULongType{2,5} ]
torch_tensor(array(runif(12), dim = c(2, 2, 3)))
#> torch_tensor
#> (1,.,.) = 
#>   0.6008  0.4978  0.8746
#>   0.0074  0.7329  0.0342
#> 
#> (2,.,.) = 
#>   0.1572  0.2898  0.1749
#>   0.4664  0.7725  0.3204
#> [ CPUFloatType{2,2,3} ]
```

By default, we will create tensors in the `cpu` device, converting their
R datatype to the corresponding torch `dtype`.

> **Note** currently, only numeric and boolean types are supported.

You can always modify `dtype` and `device` when converting an R object
to a torch tensor. For example:

``` r
torch_tensor(1, dtype = torch_long())
#> torch_tensor
#>  1
#> [ CPULongType{1} ]
torch_tensor(1, device = "cpu", dtype = torch_float64())
#> torch_tensor
#>  1
#> [ CPUDoubleType{1} ]
```

Other options available when creating a tensor are:

- `requires_grad`: boolean indicating if you want `autograd` to record
  operations on them for automatic differentiation.
- `pin_memory`: – If set, the tensor returned would be allocated in
  pinned memory. Works only for CPU tensors.

These options are available for all functions that can be used to create
new tensors, including the factory functions listed in the next section.

## Using creation functions

You can also use the `torch_*` functions listed below to create torch
tensors using some algorithm.

For example, the `torch_randn` function will create tensors using the
normal distribution with mean 0 and standard deviation 1. You can use
the `...` argument to pass the size of the dimensions. For example, the
code below will create a normally distributed tensor with shape 5x3.

``` r
x <- torch_randn(5, 3)
x
#> torch_tensor
#> -0.2707 -0.4625  1.3138
#> -1.4071  0.0370 -0.4536
#> -0.0252  0.6134  0.2075
#> -0.3414  0.5360  1.1105
#>  0.2677 -0.5010  0.3346
#> [ CPUFloatType{5,3} ]
```

Another example is `torch_ones`, which creates a tensor filled with
ones.

``` r
x <- torch_ones(2, 4, dtype = torch_int64(), device = "cpu")
x
#> torch_tensor
#>  1  1  1  1
#>  1  1  1  1
#> [ CPULongType{2,4} ]
```

Here is the full list of functions that can be used to bulk-create
tensors in torch:

- `torch_arange`: Returns a tensor with a sequence of integers,
- `torch_empty`: Returns a tensor with uninitialized values,
- `torch_eye`: Returns an identity matrix,
- `torch_full`: Returns a tensor filled with a single value,
- `torch_linspace`: Returns a tensor with values linearly spaced in some
  interval,
- `torch_logspace`: Returns a tensor with values logarithmically spaced
  in some interval,
- `torch_ones`: Returns a tensor filled with all ones,
- `torch_rand`: Returns a tensor filled with values drawn from a uniform
  distribution on \[0, 1).
- `torch_randint`: Returns a tensor with integers randomly drawn from an
  interval,
- `torch_randn`: Returns a tensor filled with values drawn from a unit
  normal distribution,
- `torch_randperm`: Returns a tensor filled with a random permutation of
  integers in some interval,
- `torch_zeros`: Returns a tensor filled with all zeros.

## Conversion

Once a tensor exists you can convert between `dtype`s and move to a
different device with `to` method. For example:

``` r
x <- torch_tensor(1)
y <- x$to(dtype = torch_int32())
x
#> torch_tensor
#>  1
#> [ CPUFloatType{1} ]
y
#> torch_tensor
#>  1
#> [ CPUIntType{1} ]
```

You can also copy a tensor to the GPU using:

``` r
x <- torch_tensor(1)
y <- x$cuda())
```
