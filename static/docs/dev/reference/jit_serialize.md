# Serialize a Script Module

Serializes a script module and returns it as a raw vector. You can read
the object again using
[`jit_unserialize`](https://torch.mlverse.org/docs/dev/reference/jit_unserialize.md).

## Usage

``` r
jit_serialize(obj)
```

## Arguments

- obj:

  (`script_module`)  
  Model to be serialized.

## Value

[`raw()`](https://rdrr.io/r/base/raw.html)

## Examples

``` r
if (torch_is_installed()) {
model <- jit_trace(nn_linear(1, 1), torch_randn(1))
serialized <- jit_serialize(model)
}
```
