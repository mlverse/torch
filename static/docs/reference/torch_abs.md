# Abs

Abs

## Usage

``` r
torch_abs(self)
```

## Arguments

- self:

  (Tensor) the input tensor.

## abs(input) -\> Tensor

Computes the element-wise absolute value of the given `input` tensor.

\$\$ \mbox{out}\_{i} = \|\mbox{input}\_{i}\| \$\$

## Examples

``` r
if (torch_is_installed()) {

torch_abs(torch_tensor(c(-1, -2, 3)))
}
#> torch_tensor
#>  1
#>  2
#>  3
#> [ CPUFloatType{3} ]
```
