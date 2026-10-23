# Abstract base class for constraints.

Abstract base class for constraints.

Abstract base class for constraints.

## Details

A constraint object represents a region over which a variable is valid,
e.g. within which a variable can be optimized.

## Methods

### Public methods

- [`Constraint$check()`](#method-torch_Constraint-check)

- [`Constraint$print()`](#method-torch_Constraint-print)

- [`Constraint$clone()`](#method-torch_Constraint-clone)

------------------------------------------------------------------------

### Method `check()`

Returns a byte tensor of `sample_shape + batch_shape` indicating whether
each event in value satisfies this constraint.

#### Usage

    Constraint$check(value)

#### Arguments

- `value`:

  each event in value will be checked.

------------------------------------------------------------------------

### Method [`print()`](https://rdrr.io/r/base/print.html)

Define the print method for constraints,

#### Usage

    Constraint$print()

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    Constraint$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
