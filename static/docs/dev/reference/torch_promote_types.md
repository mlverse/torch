# Promote_types

Promote_types

## Usage

``` r
torch_promote_types(type1, type2)
```

## Arguments

- type1:

  (`torch.dtype`)

- type2:

  (`torch.dtype`)

## promote_types(type1, type2) -\> dtype

Returns the `torch_dtype` with the smallest size and scalar kind that is
not smaller nor of lower kind than either `type1` or `type2`. See type
promotion documentation for more information on the type promotion
logic.

## Examples

``` r
if (torch_is_installed()) {

torch_promote_types(torch_int32(), torch_float32())
torch_promote_types(torch_uint8(), torch_long())
}
#> torch_Long
```
