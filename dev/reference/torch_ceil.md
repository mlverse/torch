# Ceil

Ceil

## Usage

``` r
torch_ceil(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## ceil(input, out=NULL) -\> Tensor

Returns a new tensor with the ceil of the elements of `input`, the
smallest integer greater than or equal to each element.

\$\$ \mbox{out}\_{i} = \left\lceil \mbox{input}\_{i} \right\rceil =
\left\lfloor \mbox{input}\_{i} \right\rfloor + 1 \$\$

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_ceil(a)
}
#> torch_tensor
#> -0
#>  2
#> -0
#>  2
#> [ CPUFloatType{4} ]
```
