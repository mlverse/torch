# Count_nonzero

Count_nonzero

## Usage

``` r
torch_count_nonzero(self, dim = NULL)
```

## Arguments

- self:

  (Tensor) the input tensor.

- dim:

  (int or tuple of ints, optional) Dim or tuple of dims along which to
  count non-zeros.

## count_nonzero(input, dim=None) -\> Tensor

Counts the number of non-zero values in the tensor `input` along the
given `dim`. If no dim is specified then all non-zeros in the tensor are
counted.

## Examples

``` r
if (torch_is_installed()) {

x <- torch_zeros(3,3)
x[torch_randn(3,3) > 0.5] = 1
x
torch_count_nonzero(x)
torch_count_nonzero(x, dim=1)
}
#> torch_tensor
#>  3
#>  0
#>  0
#> [ CPULongType{3} ]
```
