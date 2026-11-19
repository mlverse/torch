# Ldexp

Ldexp

## Usage

``` r
torch_ldexp(self, other)
```

## Arguments

- self:

  (Tensor) the tensor of mantissas

- other:

  (Tensor) the tensor of exponents, must be an integer dtype

## ldexp(input, other, out=NULL) -\> Tensor

Multiplies `input` by \\2^{other}\\.

\$\$ \text{out}\_i = \text{input}\_i \* 2^{\text{other}\_i} \$\$

Typically this function is used to construct floating point numbers by
multiplying mantissas in `input` with integral powers of two created
from the exponents in `other`.
