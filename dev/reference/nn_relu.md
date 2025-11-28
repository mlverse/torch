# ReLU module

Applies the rectified linear unit function element-wise
\$\$\mbox{ReLU}(x) = (x)^+ = \max(0, x)\$\$

## Usage

``` r
nn_relu(inplace = FALSE)
```

## Arguments

- inplace:

  can optionally do the operation in-place. Default: `FALSE`

## Shape

- Input: \\(N, \*)\\ where `*` means, any number of additional
  dimensions

- Output: \\(N, \*)\\, same shape as the input

## Examples

``` r
if (torch_is_installed()) {
m <- nn_relu()
input <- torch_randn(2)
m(input)
}
#> torch_tensor
#>  1.2114
#>  0.0000
#> [ CPUFloatType{2} ]
```
