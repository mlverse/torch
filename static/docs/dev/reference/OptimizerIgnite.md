# Abstract Base Class for LibTorch Optimizers

Abstract base class for wrapping LibTorch C++ optimizers.

## Super class

`torch::torch_optimizer` -\> `OptimizerIgnite`

## Methods

### Public methods

- [`OptimizerIgnite$new()`](#method-OptimizerIgnite-new)

- [`OptimizerIgnite$state_dict()`](#method-OptimizerIgnite-state_dict)

- [`OptimizerIgnite$load_state_dict()`](#method-OptimizerIgnite-load_state_dict)

- [`OptimizerIgnite$step()`](#method-OptimizerIgnite-step)

- [`OptimizerIgnite$zero_grad()`](#method-OptimizerIgnite-zero_grad)

- [`OptimizerIgnite$add_param_group()`](#method-OptimizerIgnite-add_param_group)

- [`OptimizerIgnite$clone()`](#method-OptimizerIgnite-clone)

------------------------------------------------------------------------

### Method `new()`

Initializes the optimizer with the specified parameters and defaults.

#### Usage

    OptimizerIgnite$new(params, defaults)

#### Arguments

- `params`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  Either a list of tensors or a list of parameter groups, each
  containing the `params` to optimizer as well as the optimizer options
  such as the learning rate, weight decay, etc.

- `defaults`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  A list of default optimizer options.

------------------------------------------------------------------------

### Method `state_dict()`

Returns the state dictionary containing the current state of the
optimizer. The returned [`list()`](https://rdrr.io/r/base/list.html)
contains two lists:

- `param_groups`: The parameter groups of the optimizer (`lr`, ...) as
  well as to which parameters they are applied (`params`, integer
  indices)

- `state`: The states of the optimizer. The names are the indices of the
  parameters to which they belong, converted to character.

#### Usage

    OptimizerIgnite$state_dict()

#### Returns

([`list()`](https://rdrr.io/r/base/list.html))

------------------------------------------------------------------------

### Method [`load_state_dict()`](https://torch.mlverse.org/docs/dev/reference/load_state_dict.md)

Loads the state dictionary into the optimizer.

#### Usage

    OptimizerIgnite$load_state_dict(state_dict)

#### Arguments

- `state_dict`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  The state dictionary to load into the optimizer.

------------------------------------------------------------------------

### Method [`step()`](https://rdrr.io/r/stats/step.html)

Performs a single optimization step.

#### Usage

    OptimizerIgnite$step(closure = NULL)

#### Arguments

- `closure`:

  (`function()`)  
  A closure that conducts the forward pass and returns the loss.

#### Returns

([`numeric()`](https://rdrr.io/r/base/numeric.html))  
The loss.

------------------------------------------------------------------------

### Method `zero_grad()`

Zeros out the gradients of the parameters.

#### Usage

    OptimizerIgnite$zero_grad()

------------------------------------------------------------------------

### Method `add_param_group()`

Adds a new parameter group to the optimizer.

#### Usage

    OptimizerIgnite$add_param_group(param_group)

#### Arguments

- `param_group`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  A parameter group to add to the optimizer. This should contain the
  `params` to optimize as well as the optimizer options. For all options
  that are not specified, the defaults are used.

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    OptimizerIgnite$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
