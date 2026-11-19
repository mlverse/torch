# Block_diag

Create a block diagonal matrix from provided tensors.

## Usage

``` r
torch_block_diag(tensors)
```

## Arguments

- tensors:

  (list of tensors) One or more tensors with 0, 1, or 2 dimensions.

## Examples

``` r
if (torch_is_installed()) {

A <- torch_tensor(rbind(c(0, 1), c(1, 0)))
B <- torch_tensor(rbind(c(3, 4, 5), c(6, 7, 8)))
C <- torch_tensor(7)
D <- torch_tensor(c(1, 2, 3))
E <- torch_tensor(rbind(4, 5, 6))
torch_block_diag(list(A, B, C, D, E))
}
#> torch_tensor
#>  0  1  0  0  0  0  0  0  0  0
#>  1  0  0  0  0  0  0  0  0  0
#>  0  0  3  4  5  0  0  0  0  0
#>  0  0  6  7  8  0  0  0  0  0
#>  0  0  0  0  0  7  0  0  0  0
#>  0  0  0  0  0  0  1  2  3  0
#>  0  0  0  0  0  0  0  0  0  4
#>  0  0  0  0  0  0  0  0  0  5
#>  0  0  0  0  0  0  0  0  0  6
#> [ CPUFloatType{9,10} ]
```
