#' @include utils.R

#' Generic R6 class representing distributions
#'
#' @name Distribution
#' @title Generic R6 class representing distributions
#' @rdname Distribution
#'
#' @description
#' Distribution is the abstract base class for probability distributions.
#' Note: in Python, adding torch.Size objects works as concatenation
#' Try for example: torch.Size((2, 1)) + torch.Size((1,))
#'
#' @param value values to evaluate the density on.
#' @param sample_shape the shape you want to sample.

Distribution <- R6::R6Class(
  "torch_Distribution",
  lock_objects = FALSE,
  public = list(

    #' @field .validate_args whether to validate arguments
    .validate_args = FALSE,

    #' @field has_rsample whether has an rsample
    has_rsample = FALSE,

    #' @field has_enumerate_support whether has enumerate support
    has_enumerate_support = FALSE,

    #' @description
    #' Initializes a distribution class.
    #'
    #' @param batch_shape the shape over which parameters are batched.
    #' @param event_shape the shape of a single sample (without batching).
    #' @param validate_args whether to validate the arguments or not. Validation
    #'   can be time consuming so you might want to disable it.
    initialize = function(batch_shape = NULL, event_shape = NULL, validate_args = NULL) {
      private$.batch_shape <- batch_shape
      private$.event_shape <- event_shape

      if (!is.null(validate_args)) {
        self$.validate_args <- validate_args
      }

      for (param in names(private$.arg_constraints)) {
        constraint <- private$.arg_constraints[[param]]

        if (is_dependent(constraint)) {
          next
        }

        # TODO: check lazy_property
        # if (!(param %in% names(self) && inherits(constraint, "lazy_property"))
        #   next # skip checking lazily-constructed args
        # if (all(constraint$check(constraint)))
        #   value_error("The parameter {param} has invalid values")
      }
    },

    #' @description
    #' Returns a new distribution instance (or populates an existing instance
    #' provided by a derived class) with batch dimensions expanded to batch_shape.
    #' This method calls expand on the distribution’s parameters. As such, this
    #' does not allocate new memory for the expanded distribution instance.
    #' Additionally, this does not repeat any args checking or parameter
    #' broadcasting in `initialize`, when an instance is first created.
    #'
    #' @param batch_shape the desired expanded size.
    #' @param .instance new instance provided by subclasses that need to
    #'   override `expand`.
    #'
    expand = function(batch_shape, .instance = NULL) {
      not_implemented_error()
    },

    #' @description
    #' Generates a `sample_shape` shaped sample or `sample_shape` shaped batch of
    #' samples if the distribution parameters are batched.
    #'
    sample = function(sample_shape = NULL) {
      with_no_grad({
        self$rsample(sample_shape)
      })
    },

    #' @description
    #' Generates a sample_shape shaped reparameterized sample or sample_shape
    #' shaped batch of reparameterized samples if the distribution parameters
    #' are batched.
    #'
    rsample = function(sample_shape = NULL) {
      not_implemented_error()
    },

    #' @description
    #' Returns the log of the probability density/mass function evaluated at
    #' `value`.
    #'
    log_prob = function(value) {
      not_implemented_error()
    },

    #' @description
    #' Returns the cumulative density/mass function evaluated at
    #' `value`.
    #'
    cdf = function(value) {
      not_implemented_error()
    },

    #' @description
    #' Returns the inverse cumulative density/mass function evaluated at
    #' `value`.
    #'
    icdf = function(value) {
      not_implemented_error()
    },

    #'  @description
    #' Returns tensor containing all values supported by a discrete
    #' distribution. The result will enumerate over dimension 0, so the shape
    #' of the result will be `(cardinality,) + batch_shape + event_shape
    #' (where `event_shape = ()` for univariate distributions).
    #' Note that this enumerates over all batched tensors in lock-step
    #' `list(c(0, 0), c(1, 1), ...)`. With `expand=FALSE`, enumeration happens
    #' along dim 0, but with the remaining batch dimensions being
    #' singleton dimensions, `list(c(0), c(1), ...)`.
    #' @param expand (bool): whether to expand the support over the
    #'  batch dims to match the distribution's `batch_shape`.
    #' @return Tensor iterating over dimension 0.
    enumerate_support = function(expand = TRUE) {
      not_implemented_error()
    },

    #' @description
    #' Returns entropy of distribution, batched over batch_shape.
    #' @return Tensor of shape batch_shape.
    entropy = function() {
      not_implemented_error()
    },

    #' @description
    #' Returns perplexity of distribution, batched over batch_shape.
    #' @return Tensor of shape batch_shape.
    perplexity = function() {
      not_implemented_error()
    },

    #' @description
    #' Returns the size of the sample returned by the distribution, given
    #' a `sample_shape`. Note, that the batch and event shapes of a distribution
    #' instance are fixed at the time of construction. If this is empty, the
    #' returned shape is upcast to (1,).
    #' @param sample_shape (torch_Size): the size of the sample to be drawn.
    .extended_shape = function(sample_shape = NULL) {
      c(sample_shape, self$batch_shape, self$event_shape)
    },

    #' @description
    #' Argument validation for distribution methods such as `log_prob`,
    #' `cdf` and `icdf`. The rightmost dimensions of a value to be
    #' scored via these methods must agree with the distribution's batch
    #' and event shapes.
    #' @param value (Tensor): the tensor whose log probability is to be
    #' computed by the `log_prob` method.
    .validate_sample = function(value) {
      if (!inherits(value, "torch_tensor")) {
        value_error("The value argument to log_prob must be a Tensor")
      }

      event_dim_start <- length(value$size()) - length(private$.event_shape)

      if (value$size()[event_dim_start, ] != private$.event_shape) {
        value_error(
          "The right-most size of value must match event_shape:
           {value$size()} vs {private$.event_shape}."
        )
      }

      actual_shape <- value$size()
      expected_shape <- c(private$.batch_shape, private$.event_shape)

      shape_length <- length(actual_shape)

      for (idx in shape_length:1) {
        i <- actual_shape[idx]
        j <- expected_shape[idx]

        if (i != 1 && j != 1 && i != j) {
          value_error(
            "Value is not broadcastable with
             batch_shape+event_shape: {actual_shape} vs {expected_shape}."
          )
        }
      }

      if (!self$support$check(value)$all()) {
        value_error("The value argument must be within the support")
      }
    },

    #' @description
    #' Prints the distribution instance.
    print = function() {
      param_names <-
        names(private$.arg_constraints)[
          names(private$.arg_constraints) %in% names(self)
        ]

      args_string <- paste(
        param_names, sapply(param_names, function(x) {
          x <- self[[x]]
          as.array(if (is.function(x)) x() else x)
        }),
        sep = "=", collapse = ", "
      )

      class_name <- class(self)[1]
      cat(glue::glue("{class_name} ({args_string})"))
    }
  ),
  active = list(

    #' @field batch_shape Returns the shape over which parameters are batched.
    batch_shape = function() {
      private$.batch_shape
    },

    #' @field event_shape Returns the shape of a single sample (without batching).
    event_shape = function() {
      private$.event_shape
    },

    #' Returns a dictionary from argument names to
    #' `torch_Constraint` objects that
    #' should be satisfied by each argument of this distribution. Args that
    #' are not tensors need not appear in this dict.
    # arg_constraints = function(){
    #   not_implemented_error()
    # },

    #' @field support Returns a `torch_Constraint` object representing this distribution's
    #'   support.
    support = function() {
      not_implemented_error()
    },

    #' @field mean Returns the mean on of the distribution
    mean = function() {
      not_implemented_error()
    },

    #' @field variance Returns the variance of the distribution
    variance = function() {
      not_implemented_error()
    },

    #' @field stddev Returns the standard deviation of the distribution
    stddev = function() {
      self$variance$sqrt()
    }
  ),
  private = list(
    .support = NULL,
    .batch_shape = NULL,
    .event_shape = NULL,

    # Choose different structure?
    .arg_constraints = list(),
    .get_checked_instance = function(cls, .instance = NULL, .args) {
      if (is.null(.instance) && !identical(self$initialize, cls$initialize)) {
        #' TODO: consider different message
        not_implemented_error(
          "Subclass {paste0(class(self), collapse = ' ')} of ",
          "{paste0(class(cls), collapse = ' ')} ",
          "that defines a custom `initialize()` method ",
          "must also define a custom `expand()` method."
        )
      }

      if (is.null(.instance)) {
        return(do.call(self$class_def$new, .args))
      } else {
        return(.instance)
      }
    }
  )
)

Distribution <- add_class_definition(Distribution)
