# Floor

Floor

## Usage

``` r
torch_floor(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## floor(input, out=NULL) -\> Tensor

Returns a new tensor with the floor of the elements of `input`, the
largest integer less than or equal to each element.

\$\$ \mbox{out}\_{i} = \left\lfloor \mbox{input}\_{i} \right\rfloor \$\$

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_floor(a)
}
#> torch_tensor
#> -2
#>  0
#>  2
#> -2
#> [ CPUFloatType{4} ]
```
