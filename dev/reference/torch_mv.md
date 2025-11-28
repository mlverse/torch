# Mv

Mv

## Usage

``` r
torch_mv(self, vec)
```

## Arguments

- self:

  (Tensor) matrix to be multiplied

- vec:

  (Tensor) vector to be multiplied

## Note

This function does not broadcast .

## mv(input, vec, out=NULL) -\> Tensor

Performs a matrix-vector product of the matrix `input` and the vector
`vec`.

If `input` is a \\(n \times m)\\ tensor, `vec` is a 1-D tensor of size
\\m\\, `out` will be 1-D of size \\n\\.

## Examples

``` r
if (torch_is_installed()) {

mat = torch_randn(c(2, 3))
vec = torch_randn(c(3))
torch_mv(mat, vec)
}
#> torch_tensor
#>  1.7796
#> -0.9408
#> [ CPUFloatType{2} ]
```
