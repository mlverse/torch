# Sqrt

Sqrt

## Usage

``` r
torch_sqrt(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## sqrt(input, out=NULL) -\> Tensor

Returns a new tensor with the square-root of the elements of `input`.

\$\$ \mbox{out}\_{i} = \sqrt{\mbox{input}\_{i}} \$\$

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_sqrt(a)
}
#> torch_tensor
#>  1.3328
#>     nan
#>     nan
#>  0.9272
#> [ CPUFloatType{4} ]
```
