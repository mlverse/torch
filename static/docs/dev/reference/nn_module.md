# Base class for all neural network modules.

Your models should also subclass this class.

## Usage

``` r
nn_module(
  classname = NULL,
  inherit = nn_Module,
  ...,
  private = NULL,
  active = NULL,
  parent_env = parent.frame()
)
```

## Arguments

- classname:

  an optional name for the module

- inherit:

  an optional module to inherit from

- ...:

  methods implementation

- private:

  passed to
  [`R6::R6Class()`](https://r6.r-lib.org/reference/R6Class.html).

- active:

  passed to
  [`R6::R6Class()`](https://r6.r-lib.org/reference/R6Class.html).

- parent_env:

  passed to
  [`R6::R6Class()`](https://r6.r-lib.org/reference/R6Class.html).

## Details

Modules can also contain other Modules, allowing to nest them in a tree
structure. You can assign the submodules as regular attributes.

You are expected to implement the `initialize` and the `forward` to
create a new `nn_module`.

## Initialize

The initialize function will be called whenever a new instance of the
`nn_module` is created. We use the initialize functions to define
submodules and parameters of the module. For example:

    initialize = function(input_size, output_size) {
       self$conv1 <- nn_conv2d(input_size, output_size, 5)
       self$conv2 <- nn_conv2d(output_size, output_size, 5)
    }

The initialize function can have any number of parameters. All objects
assigned to `self$` will be available for other methods that you
implement. Tensors wrapped with
[`nn_parameter()`](https://torch.mlverse.org/docs/dev/reference/nn_parameter.md)
or
[`nn_buffer()`](https://torch.mlverse.org/docs/dev/reference/nn_buffer.md)
and submodules are automatically tracked when assigned to `self$`.

The initialize function is optional if the module you are defining
doesn't have weights, submodules or buffers.

## Forward

The forward method is called whenever an instance of `nn_module` is
called. This is usually used to implement the computation that the
module does with the weights ad submodules defined in the `initialize`
function.

For example:

    forward = function(input) {
       input <- self$conv1(input)
       input <- nnf_relu(input)
       input <- self$conv2(input)
       input <- nnf_relu(input)
       input
     }

The `forward` function can use the `self$training` attribute to make
different computations depending wether the model is training or not,
for example if you were implementing the dropout module.

## Cloning

To finalize the cloning of a module, you can define a private
`finalize_deep_clone()` method. This method is called on the cloned
object when deep-cloning a module, after all the modules, parameters and
buffers were already cloned.

## Examples

``` r
if (torch_is_installed()) {
model <- nn_module(
  initialize = function() {
    self$conv1 <- nn_conv2d(1, 20, 5)
    self$conv2 <- nn_conv2d(20, 20, 5)
  },
  forward = function(input) {
    input <- self$conv1(input)
    input <- nnf_relu(input)
    input <- self$conv2(input)
    input <- nnf_relu(input)
    input
  }
)
}
```
