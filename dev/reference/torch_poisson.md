# Poisson

Poisson

## Usage

``` r
torch_poisson(self, generator = NULL)
```

## Arguments

- self:

  (Tensor) the input tensor containing the rates of the Poisson
  distribution

- generator:

  (`torch.Generator`, optional) a pseudorandom number generator for
  sampling

## poisson(input \*, generator=NULL) -\> Tensor

Returns a tensor of the same size as `input` with each element sampled
from a Poisson distribution with rate parameter given by the
corresponding element in `input` i.e.,

\$\$ \mbox{out}\_i \sim \mbox{Poisson}(\mbox{input}\_i) \$\$

## Examples

``` r
if (torch_is_installed()) {

rates = torch_rand(c(4, 4)) * 5  # rate parameter between 0 and 5
torch_poisson(rates)
}
#> torch_tensor
#>  2  1  6  6
#>  8  5  3  0
#>  0  0  2  4
#>  1  0  0  0
#> [ CPUFloatType{4,4} ]
```
