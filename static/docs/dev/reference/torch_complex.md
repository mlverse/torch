# Complex

Complex

## Usage

``` r
torch_complex(real, imag)
```

## Arguments

- real:

  (Tensor) The real part of the complex tensor. Must be float or double.

- imag:

  (Tensor) The imaginary part of the complex tensor. Must be same dtype
  as `real`.

## complex(real, imag, \*, out=None) -\> Tensor

Constructs a complex tensor with its real part equal to `real` and its
imaginary part equal to `imag`.

## Examples

``` r
if (torch_is_installed()) {

real <- torch_tensor(c(1, 2), dtype=torch_float32())
imag <- torch_tensor(c(3, 4), dtype=torch_float32())
z <- torch_complex(real, imag)
z
z$dtype
}
#> torch_ComplexFloat
```
