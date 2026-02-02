# Renorm

Renorm

## Usage

``` r
torch_renorm(self, p, dim, maxnorm)
```

## Arguments

- self:

  (Tensor) the input tensor.

- p:

  (float) the power for the norm computation

- dim:

  (int) the dimension to slice over to get the sub-tensors

- maxnorm:

  (float) the maximum norm to keep each sub-tensor under

## Note

If the norm of a row is lower than `maxnorm`, the row is unchanged

## renorm(input, p, dim, maxnorm, out=NULL) -\> Tensor

Returns a tensor where each sub-tensor of `input` along dimension `dim`
is normalized such that the `p`-norm of the sub-tensor is lower than the
value `maxnorm`

## Examples

``` r
if (torch_is_installed()) {
x = torch_ones(c(3, 3))
x[2,]$fill_(2)
x[3,]$fill_(3)
x
torch_renorm(x, 1, 1, 5)
}
#> torch_tensor
#>  1.0000  1.0000  1.0000
#>  1.6667  1.6667  1.6667
#>  1.6667  1.6667  1.6667
#> [ CPUFloatType{3,3} ]
```
