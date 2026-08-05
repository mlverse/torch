# nn_utils_weight_norm

Applies weight normalization to a parameter in the given module.

## Value

The original module with the weight_v and weight_g paramters.

## Details

        \eqn{\mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|}}

Weight normalization is a reparameterization that decouples the
magnitude of a weight tensor from its direction. This replaces the
parameter specified by `name` (e.g. `'weight'`) with two parameters: one
specifying the magnitude (e.g. `'weight_g'`) and one specifying the
direction (e.g. `'weight_v'`).

## Note

The pytorch Weight normalization is implemented via a hook that
recomputes the weight tensor from the magnitude and direction before
every `forward()` call. Since torch for R still do not support hooks,
the weight recomputation need to be done explicitly inside the
`forward()` definition trough a call of the `recompute()` method. See
examples.

By default, with `dim = 0`, the norm is computed independently per
output channel/plane. To compute a norm over the entire weight tensor,
use `dim = NULL`.

@references https://arxiv.org/abs/1602.07868

## Methods

### Public methods

- [`nn_utils_weight_norm$new()`](#method-nn_utils_weight_norm-new)

- [`nn_utils_weight_norm$compute_weight()`](#method-nn_utils_weight_norm-compute_weight)

- [`nn_utils_weight_norm$apply()`](#method-nn_utils_weight_norm-apply)

- [`nn_utils_weight_norm$call()`](#method-nn_utils_weight_norm-call)

- [`nn_utils_weight_norm$recompute()`](#method-nn_utils_weight_norm-recompute)

- [`nn_utils_weight_norm$remove()`](#method-nn_utils_weight_norm-remove)

- [`nn_utils_weight_norm$clone()`](#method-nn_utils_weight_norm-clone)

------------------------------------------------------------------------

### Method `new()`

#### Usage

    nn_utils_weight_norm$new(name, dim)

#### Arguments

- `name`:

  (str, optional): name of weight parameter

- `dim`:

  (int, optional): dimension over which to compute the norm

------------------------------------------------------------------------

### Method `compute_weight()`

#### Usage

    nn_utils_weight_norm$compute_weight(module, name = NULL, dim = NULL)

#### Arguments

- `module`:

  (Module): containing module

- `name`:

  (str, optional): name of weight parameter

- `dim`:

  (int, optional): dimension over which to compute the norm

------------------------------------------------------------------------

### Method [`apply()`](https://rdrr.io/r/base/apply.html)

#### Usage

    nn_utils_weight_norm$apply(module, name = NULL, dim = NULL)

#### Arguments

- `module`:

  (Module): containing module

- `name`:

  (str, optional): name of weight parameter

- `dim`:

  (int, optional): dimension over which to compute the norm

------------------------------------------------------------------------

### Method [`call()`](https://rdrr.io/r/base/call.html)

#### Usage

    nn_utils_weight_norm$call(module)

#### Arguments

- `module`:

  (Module): containing module

------------------------------------------------------------------------

### Method `recompute()`

#### Usage

    nn_utils_weight_norm$recompute(module)

#### Arguments

- `module`:

  (Module): containing module

------------------------------------------------------------------------

### Method [`remove()`](https://rdrr.io/r/base/rm.html)

#### Usage

    nn_utils_weight_norm$remove(module, name = NULL)

#### Arguments

- `module`:

  (Module): containing module

- `name`:

  (str, optional): name of weight parameter

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    nn_utils_weight_norm$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
if (torch_is_installed()) {
x = nn_linear(in_features = 20, out_features = 40)
weight_norm = nn_utils_weight_norm$new(name = 'weight', dim = 2)
weight_norm$apply(x)
x$weight_g$size()
x$weight_v$size()
x$weight

# the recompute() method recomputes the weight using g and v. It must be called
# explicitly inside `forward()`.
weight_norm$recompute(x)

}
```
