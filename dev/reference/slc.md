# Creates a slice

Creates a slice object that can be used when indexing torch tensors.

## Usage

``` r
slc(start, end, step = 1)
```

## Arguments

- start:

  (integer) starting index.

- end:

  (integer) the last selected index.

- step:

  (integer) the step between indexes.

## Examples

``` r
if (torch_is_installed()) {
x <- torch_randn(10)
x[slc(start = 1, end = 5, step = 2)]

}
#> torch_tensor
#>  0.4246
#>  0.7825
#> -1.2273
#> [ CPUFloatType{3} ]
```
