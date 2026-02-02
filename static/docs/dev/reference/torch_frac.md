# Frac

Frac

## Usage

``` r
torch_frac(self)
```

## Arguments

- self:

  the input tensor.

## frac(input, out=NULL) -\> Tensor

Computes the fractional portion of each element in `input`.

\$\$ \mbox{out}\_{i} = \mbox{input}\_{i} - \left\lfloor
\|\mbox{input}\_{i}\| \right\rfloor \* \mbox{sgn}(\mbox{input}\_{i})
\$\$

## Examples

``` r
if (torch_is_installed()) {

torch_frac(torch_tensor(c(1, 2.5, -3.2)))
}
#> torch_tensor
#>  0.0000
#>  0.5000
#> -0.2000
#> [ CPUFloatType{3} ]
```
