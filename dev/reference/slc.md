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
#> -1.0529
#>  0.1245
#> -0.1684
#> [ CPUFloatType{3} ]
```
