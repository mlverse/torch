# A sequential container

A sequential container. Modules will be added to it in the order they
are passed in the constructor. See examples.

## Usage

``` r
nn_sequential(...)
```

## Arguments

- ...:

  sequence of modules to be added

## Examples

``` r
if (torch_is_installed()) {

model <- nn_sequential(
  nn_conv2d(1, 20, 5),
  nn_relu(),
  nn_conv2d(20, 64, 5),
  nn_relu()
)
input <- torch_randn(32, 1, 28, 28)
output <- model(input)
}
```
