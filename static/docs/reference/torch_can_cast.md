# Can_cast

Can_cast

## Usage

``` r
torch_can_cast(from_, to)
```

## Arguments

- from\_:

  (dtype) The original `torch_dtype`.

- to:

  (dtype) The target `torch_dtype`.

## can_cast(from, to) -\> bool

Determines if a type conversion is allowed under PyTorch casting rules
described in the type promotion documentation .

## Examples

``` r
if (torch_is_installed()) {

torch_can_cast(torch_double(), torch_float())
torch_can_cast(torch_float(), torch_int())
}
#> [1] FALSE
```
