# Normal

Normal

Normal distributed

## Usage

``` r
torch_normal(mean, std, size = NULL, generator = NULL, ...)
```

## Arguments

- mean:

  (tensor or scalar double) Mean of the normal distribution. If this is
  a
  [`torch_tensor()`](https://torch.mlverse.org/docs/dev/reference/torch_tensor.md)
  then the output has the same dim as `mean` and it represents the
  per-element mean. If it's a scalar value, it's reused for all
  elements.

- std:

  (tensor or scalar double) The standard deviation of the normal
  distribution. If this is a
  [`torch_tensor()`](https://torch.mlverse.org/docs/dev/reference/torch_tensor.md)
  then the output has the same size as `std` and it represents the
  per-element standard deviation. If it's a scalar value, it's reused
  for all elements.

- size:

  (integers, optional) only used if both `mean` and `std` are scalars.

- generator:

  a random number generator created with
  [`torch_generator()`](https://torch.mlverse.org/docs/dev/reference/torch_generator.md).
  If `NULL` a default generator is used.

- ...:

  Tensor option parameters like `dtype`, `layout`, and `device`. Can
  only be used when `mean` and `std` are both scalar numerics.

## Note

When the shapes do not match, the shape of `mean` is used as the shape
for the returned output tensor

## normal(mean, std, \*) -\> Tensor

Returns a tensor of random numbers drawn from separate normal
distributions whose mean and standard deviation are given.

The `mean` is a tensor with the mean of each output element's normal
distribution

The `std` is a tensor with the standard deviation of each output
element's normal distribution

The shapes of `mean` and `std` don't need to match, but the total number
of elements in each tensor need to be the same.

## normal(mean=0.0, std) -\> Tensor

Similar to the function above, but the means are shared among all drawn
elements.

## normal(mean, std=1.0) -\> Tensor

Similar to the function above, but the standard-deviations are shared
among all drawn elements.

## normal(mean, std, size, \*) -\> Tensor

Similar to the function above, but the means and standard deviations are
shared among all drawn elements. The resulting tensor has size given by
`size`.

## Examples

``` r
if (torch_is_installed()) {

torch_normal(mean=0, std=torch_arange(1, 0, -0.1) + 1e-6)
torch_normal(mean=0.5, std=torch_arange(1., 6.))
torch_normal(mean=torch_arange(1., 6.))
torch_normal(2, 3, size=c(1, 4))

}
#> torch_tensor
#>  1.9239  1.1585  2.1667  2.6931
#> [ CPUFloatType{1,4} ]
```
