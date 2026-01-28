# Cos

Cos

## Usage

``` r
torch_cos(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## cos(input, out=NULL) -\> Tensor

Returns a new tensor with the cosine of the elements of `input`.

\$\$ \mbox{out}\_{i} = \cos(\mbox{input}\_{i}) \$\$

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_cos(a)
}
#> torch_tensor
#> -0.1936
#>  0.6849
#>  0.9475
#>  0.5434
#> [ CPUFloatType{4} ]
```
