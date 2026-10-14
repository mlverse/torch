# Class representing the context.

Class representing the context.

Class representing the context.

## Public fields

- `ptr`:

  (Dev related) pointer to the context c++ object.

## Active bindings

- `needs_input_grad`:

  boolean listing arguments of `forward` and whether they require_grad.

- `saved_variables`:

  list of objects that were saved for backward via `save_for_backward`.

## Methods

### Public methods

- [`AutogradContext$new()`](#method-torch_autograd_context-new)

- [`AutogradContext$save_for_backward()`](#method-torch_autograd_context-save_for_backward)

- [`AutogradContext$mark_non_differentiable()`](#method-torch_autograd_context-mark_non_differentiable)

- [`AutogradContext$mark_dirty()`](#method-torch_autograd_context-mark_dirty)

- [`AutogradContext$clone()`](#method-torch_autograd_context-clone)

------------------------------------------------------------------------

### Method `new()`

(Dev related) Initializes the context. Not user related.

#### Usage

    AutogradContext$new(
      ptr,
      env,
      argument_names = NULL,
      argument_needs_grad = NULL
    )

#### Arguments

- `ptr`:

  pointer to the c++ object

- `env`:

  environment that encloses both forward and backward

- `argument_names`:

  names of forward arguments

- `argument_needs_grad`:

  whether each argument in forward needs grad.

------------------------------------------------------------------------

### Method `save_for_backward()`

Saves given objects for a future call to backward().

This should be called at most once, and only from inside the `forward()`
method.

Later, saved objects can be accessed through the `saved_variables`
attribute. Before returning them to the user, a check is made to ensure
they weren’t used in any in-place operation that modified their content.

Arguments can also be any kind of R object.

#### Usage

    AutogradContext$save_for_backward(...)

#### Arguments

- `...`:

  any kind of R object that will be saved for the backward pass. It's
  common to pass named arguments.

------------------------------------------------------------------------

### Method `mark_non_differentiable()`

Marks outputs as non-differentiable.

This should be called at most once, only from inside the `forward()`
method, and all arguments should be outputs.

This will mark outputs as not requiring gradients, increasing the
efficiency of backward computation. You still need to accept a gradient
for each output in `backward()`, but it’s always going to be a zero
tensor with the same shape as the shape of a corresponding output.

This is used e.g. for indices returned from a max Function.

#### Usage

    AutogradContext$mark_non_differentiable(...)

#### Arguments

- `...`:

  non-differentiable outputs.

------------------------------------------------------------------------

### Method `mark_dirty()`

Marks given tensors as modified in an in-place operation.

This should be called at most once, only from inside the `forward()`
method, and all arguments should be inputs.

Every tensor that’s been modified in-place in a call to `forward()`
should be given to this function, to ensure correctness of our checks.
It doesn’t matter whether the function is called before or after
modification.

#### Usage

    AutogradContext$mark_dirty(...)

#### Arguments

- `...`:

  tensors that are modified in-place.

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    AutogradContext$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
