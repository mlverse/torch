# Context-manager that enable anomaly detection for the autograd engine.

This does two things:

## Usage

``` r
with_detect_anomaly(code)
```

## Arguments

- code:

  Code that will be executed in the detect anomaly context.

## Details

- Running the forward pass with detection enabled will allow the
  backward pass to print the traceback of the forward operation that
  created the failing backward function.

- Any backward computation that generate "nan" value will raise an
  error.

## Warning

This mode should be enabled only for debugging as the different tests
will slow down your program execution.

## Examples

``` r
if (torch_is_installed()) {
x <- torch_randn(2, requires_grad = TRUE)
y <- torch_randn(1)
b <- (x^y)$sum()
y$add_(1)

try({
  b$backward()

  with_detect_anomaly({
    b$backward()
  })
})
}
```
