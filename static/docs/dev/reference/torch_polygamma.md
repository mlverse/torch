# Polygamma

Polygamma

## Usage

``` r
torch_polygamma(n, input)
```

## Arguments

- n:

  (int) the order of the polygamma function

- input:

  (Tensor) the input tensor.

## Note

    This function is not implemented for \eqn{n \geq 2}.

## polygamma(n, input, out=NULL) -\> Tensor

Computes the \\n^{th}\\ derivative of the digamma function on `input`.
\\n \geq 0\\ is called the order of the polygamma function.

\$\$ \psi^{(n)}(x) = \frac{d^{(n)}}{dx^{(n)}} \psi(x) \$\$

## Examples

``` r
if (torch_is_installed()) {
if (FALSE) { # \dontrun{
a = torch_tensor(c(1, 0.5))
torch_polygamma(1, a)
} # }
}
```
