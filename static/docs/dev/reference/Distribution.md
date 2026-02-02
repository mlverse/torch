# Generic R6 class representing distributions

Distribution is the abstract base class for probability distributions.
Note: in Python, adding torch.Size objects works as concatenation Try
for example: torch.Size((2, 1)) + torch.Size((1,))

## Public fields

- `.validate_args`:

  whether to validate arguments

- `has_rsample`:

  whether has an rsample

- `has_enumerate_support`:

  whether has enumerate support

## Active bindings

- `batch_shape`:

  Returns the shape over which parameters are batched.

- `event_shape`:

  Returns the shape of a single sample (without batching). Returns a
  dictionary from argument names to `torch_Constraint` objects that
  should be satisfied by each argument of this distribution. Args that
  are not tensors need not appear in this dict.

- `support`:

  Returns a `torch_Constraint` object representing this distribution's
  support.

- `mean`:

  Returns the mean on of the distribution

- `variance`:

  Returns the variance of the distribution

- `stddev`:

  Returns the standard deviation of the distribution TODO: consider
  different message

## Methods

### Public methods

- [`Distribution$new()`](#method-torch_Distribution-new)

- [`Distribution$expand()`](#method-torch_Distribution-expand)

- [`Distribution$sample()`](#method-torch_Distribution-sample)

- [`Distribution$rsample()`](#method-torch_Distribution-rsample)

- [`Distribution$log_prob()`](#method-torch_Distribution-log_prob)

- [`Distribution$cdf()`](#method-torch_Distribution-cdf)

- [`Distribution$icdf()`](#method-torch_Distribution-icdf)

- [`Distribution$enumerate_support()`](#method-torch_Distribution-enumerate_support)

- [`Distribution$entropy()`](#method-torch_Distribution-entropy)

- [`Distribution$perplexity()`](#method-torch_Distribution-perplexity)

- [`Distribution$.extended_shape()`](#method-torch_Distribution-.extended_shape)

- [`Distribution$.validate_sample()`](#method-torch_Distribution-.validate_sample)

- [`Distribution$print()`](#method-torch_Distribution-print)

- [`Distribution$clone()`](#method-torch_Distribution-clone)

------------------------------------------------------------------------

### Method `new()`

Initializes a distribution class.

#### Usage

    Distribution$new(batch_shape = NULL, event_shape = NULL, validate_args = NULL)

#### Arguments

- `batch_shape`:

  the shape over which parameters are batched.

- `event_shape`:

  the shape of a single sample (without batching).

- `validate_args`:

  whether to validate the arguments or not. Validation can be time
  consuming so you might want to disable it.

------------------------------------------------------------------------

### Method `expand()`

Returns a new distribution instance (or populates an existing instance
provided by a derived class) with batch dimensions expanded to
batch_shape. This method calls expand on the distribution’s parameters.
As such, this does not allocate new memory for the expanded distribution
instance. Additionally, this does not repeat any args checking or
parameter broadcasting in `initialize`, when an instance is first
created.

#### Usage

    Distribution$expand(batch_shape, .instance = NULL)

#### Arguments

- `batch_shape`:

  the desired expanded size.

- `.instance`:

  new instance provided by subclasses that need to override `expand`.

------------------------------------------------------------------------

### Method [`sample()`](https://rdrr.io/r/base/sample.html)

Generates a `sample_shape` shaped sample or `sample_shape` shaped batch
of samples if the distribution parameters are batched.

#### Usage

    Distribution$sample(sample_shape = NULL)

#### Arguments

- `sample_shape`:

  the shape you want to sample.

------------------------------------------------------------------------

### Method `rsample()`

Generates a sample_shape shaped reparameterized sample or sample_shape
shaped batch of reparameterized samples if the distribution parameters
are batched.

#### Usage

    Distribution$rsample(sample_shape = NULL)

#### Arguments

- `sample_shape`:

  the shape you want to sample.

------------------------------------------------------------------------

### Method `log_prob()`

Returns the log of the probability density/mass function evaluated at
`value`.

#### Usage

    Distribution$log_prob(value)

#### Arguments

- `value`:

  values to evaluate the density on.

------------------------------------------------------------------------

### Method `cdf()`

Returns the cumulative density/mass function evaluated at `value`.

#### Usage

    Distribution$cdf(value)

#### Arguments

- `value`:

  values to evaluate the density on.

------------------------------------------------------------------------

### Method `icdf()`

Returns the inverse cumulative density/mass function evaluated at
`value`.

@description Returns tensor containing all values supported by a
discrete distribution. The result will enumerate over dimension 0, so
the shape of the result will be
`(cardinality,) + batch_shape + event_shape (where `event_shape =
()`for univariate distributions). Note that this enumerates over all batched tensors in lock-step`list(c(0,
0), c(1, 1),
...)`. With `expand=FALSE`, enumeration happens along dim 0, but with the remaining batch dimensions being singleton dimensions, `list(c(0),
c(1), ...)\`.

#### Usage

    Distribution$icdf(value)

#### Arguments

- `value`:

  values to evaluate the density on.

------------------------------------------------------------------------

### Method `enumerate_support()`

#### Usage

    Distribution$enumerate_support(expand = TRUE)

#### Arguments

- `expand`:

  (bool): whether to expand the support over the batch dims to match the
  distribution's `batch_shape`.

#### Returns

Tensor iterating over dimension 0.

------------------------------------------------------------------------

### Method `entropy()`

Returns entropy of distribution, batched over batch_shape.

#### Usage

    Distribution$entropy()

#### Returns

Tensor of shape batch_shape.

------------------------------------------------------------------------

### Method `perplexity()`

Returns perplexity of distribution, batched over batch_shape.

#### Usage

    Distribution$perplexity()

#### Returns

Tensor of shape batch_shape.

------------------------------------------------------------------------

### Method `.extended_shape()`

Returns the size of the sample returned by the distribution, given a
`sample_shape`. Note, that the batch and event shapes of a distribution
instance are fixed at the time of construction. If this is empty, the
returned shape is upcast to (1,).

#### Usage

    Distribution$.extended_shape(sample_shape = NULL)

#### Arguments

- `sample_shape`:

  (torch_Size): the size of the sample to be drawn.

------------------------------------------------------------------------

### Method `.validate_sample()`

Argument validation for distribution methods such as `log_prob`, `cdf`
and `icdf`. The rightmost dimensions of a value to be scored via these
methods must agree with the distribution's batch and event shapes.

#### Usage

    Distribution$.validate_sample(value)

#### Arguments

- `value`:

  (Tensor): the tensor whose log probability is to be computed by the
  `log_prob` method.

------------------------------------------------------------------------

### Method [`print()`](https://rdrr.io/r/base/print.html)

Prints the distribution instance.

#### Usage

    Distribution$print()

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    Distribution$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
