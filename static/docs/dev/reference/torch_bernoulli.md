# Bernoulli

Bernoulli

## Usage

``` r
torch_bernoulli(self, p, generator = NULL)
```

## Arguments

- self:

  (Tensor) the input tensor of probability values for the Bernoulli
  distribution

- p:

  (Number) a probability value. If `p` is passed than it's used instead
  of the values in `self` tensor.

- generator:

  (`torch.Generator`, optional) a pseudorandom number generator for
  sampling

## bernoulli(input, \*, generator=NULL, out=NULL) -\> Tensor

Draws binary random numbers (0 or 1) from a Bernoulli distribution.

The `input` tensor should be a tensor containing probabilities to be
used for drawing the binary random number. Hence, all values in `input`
have to be in the range: \\0 \leq \mbox{input}\_i \leq 1\\.

The \\\mbox{i}^{th}\\ element of the output tensor will draw a value
\\1\\ according to the \\\mbox{i}^{th}\\ probability value given in
`input`.

\$\$ \mbox{out}\_{i} \sim \mathrm{Bernoulli}(p = \mbox{input}\_{i}) \$\$
The returned `out` tensor only has values 0 or 1 and is of the same
shape as `input`.

`out` can have integral `dtype`, but `input` must have floating point
`dtype`.

## Examples

``` r
if (torch_is_installed()) {

a = torch_empty(c(3, 3))$uniform_(0, 1)  # generate a uniform random matrix with range c(0, 1)
a
torch_bernoulli(a)
a = torch_ones(c(3, 3)) # probability of drawing "1" is 1
torch_bernoulli(a)
a = torch_zeros(c(3, 3)) # probability of drawing "1" is 0
torch_bernoulli(a)
}
#> torch_tensor
#>  0  0  0
#>  0  0  0
#>  0  0  0
#> [ CPUFloatType{3,3} ]
```
