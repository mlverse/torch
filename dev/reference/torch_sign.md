# Sign

Sign

## Usage

``` r
torch_sign(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## sign(input, out=NULL) -\> Tensor

Returns a new tensor with the signs of the elements of `input`.

\$\$ \mbox{out}\_{i} = \mbox{sgn}(\mbox{input}\_{i}) \$\$

## Examples

``` r
if (torch_is_installed()) {

a = torch_tensor(c(0.7, -1.2, 0., 2.3))
a
torch_sign(a)
}
#> torch_tensor
#>  1
#> -1
#>  0
#>  1
#> [ CPUFloatType{4} ]
```
