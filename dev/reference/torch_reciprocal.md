# Reciprocal

Reciprocal

## Usage

``` r
torch_reciprocal(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## reciprocal(input, out=NULL) -\> Tensor

Returns a new tensor with the reciprocal of the elements of `input`

\$\$ \mbox{out}\_{i} = \frac{1}{\mbox{input}\_{i}} \$\$

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_reciprocal(a)
}
#> torch_tensor
#> -2.1248
#> -3.3875
#> -3.2387
#>  1.1838
#> [ CPUFloatType{4} ]
```
