# Histc

Histc

## Usage

``` r
torch_histc(self, bins = 100L, min = 0L, max = 0L)
```

## Arguments

- self:

  (Tensor) the input tensor.

- bins:

  (int) number of histogram bins

- min:

  (int) lower end of the range (inclusive)

- max:

  (int) upper end of the range (inclusive)

## histc(input, bins=100, min=0, max=0, out=NULL) -\> Tensor

Computes the histogram of a tensor.

The elements are sorted into equal width bins between `min` and `max`.
If `min` and `max` are both zero, the minimum and maximum values of the
data are used.

## Examples

``` r
if (torch_is_installed()) {

torch_histc(torch_tensor(c(1., 2, 1)), bins=4, min=0, max=3)
}
#> torch_tensor
#>  0
#>  2
#>  1
#>  0
#> [ CPUFloatType{4} ]
```
