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
#> -0.1877
#>  1.2374
#>  0.4334
#> -1.3022
#> -0.3951
#> [ CPUFloatType{5} ]
```
