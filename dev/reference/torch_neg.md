# Neg

Neg

## Usage

``` r
torch_neg(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## neg(input, out=NULL) -\> Tensor

Returns a new tensor with the negative of the elements of `input`.

\$\$ \mbox{out} = -1 \times \mbox{input} \$\$

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(5))
a
torch_neg(a)
}
#> torch_tensor
#>  1.6500
#> -1.4661
#> -1.0256
#>  0.0530
#> -1.8011
#> [ CPUFloatType{5} ]
```
