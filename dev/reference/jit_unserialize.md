# Unserialize a Script Module

Unserializes a script module from a raw vector (generated with
[`jit_serialize`](https://torch.mlverse.org/docs/dev/reference/jit_serialize.md)\`).

## Usage

``` r
jit_unserialize(obj)
```

## Arguments

- obj:

  (`raw`)  
  Serialized model.

## Value

`script_module` model \<- jit_trace(nn_linear(1, 1), torch_randn(1))
serialized \<- jit_serialize(model) model2 \<-
jit_unserialize(serialized)
