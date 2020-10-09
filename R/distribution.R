#' Distribution is the abstract base class for probability distributions.
Distribution <- R6::R6Class(
  "torch_Distribution",
  lock_objects = FALSE,
  
  public = list(
    
    has_rsample           = FALSE,
    has_enumerate_support = FALSE,
    .validate_args        = FALSE,
    support               = NULL,
    
    # Choose different structure?
    arg_constraints       = list(),
    
    initialize = function(batch_shape, event_shape, validate_args = NULL){
      self$batch_shape <-  batch_shape
      self$event_shape <- event_shape
      
      if (!is.null(validate_args))
        self$validate_args <- validate_args
      
        for (p in seq_along(self$arg_constraints)) {
          constraint <- arg_constraints[[p]]$constraint
          if (constr_is_dependent(constraint))
            next
          if (!(param %in% self$.) & inherits(getattr(type(self), param), "lazy_property"))
            next # skip checking lazily-constructed args
          if (all(constr_check(getattr(self, param))))
            value_error("The parameter {param} has invalid values")
        }
    },
    
    expand = function(batch_shape, .instance = NULL){
      stop("torch_Distribution is an abstract class.")
    },
    
    #' Generates a sample_shape shaped sample or sample_shape shaped batch of
    #' samples if the distribution parameters are batched.
    sample = function(sample_shape=torch.Size()){
      with_no_grad({
        self$rsample(sample_shape)
      })
    },
    
    #' Generates a sample_shape shaped reparameterized sample or sample_shape
    #' shaped batch of reparameterized samples if the distribution parameters
    #' are batched.
    rsample = function(sample_shape=torch.Size()) NULL,
    
    #' Returns the log of the probability density/mass function evaluated at
    #' `value`.
    log_prob = function(value) NULL,
    
    #'  Returns the cumulative density/mass function evaluated at
    #' `value`.
    cdf = function(value) NULL,
    
    #'  Returns the inverse cumulative density/mass function evaluated at
    #' `value`.
    icdf = function(value) NULL,
    
    #'  Returns tensor containing all values supported by a discrete
    #'  distribution. The result will enumerate over dimension 0, so the shape
    #'  of the result will be `(cardinality,) + batch_shape + event_shape
    #'  (where `event_shape = ()` for univariate distributions).
    #'  Note that this enumerates over all batched tensors in lock-step
    #'  `[[0, 0], [1, 1], ...]`. With `expand=False`, enumeration happens
    #'  along dim 0, but with the remaining batch dimensions being
    #'  singleton dimensions, `[[0], [1], ..`.
    #'  To iterate over the full Cartesian product use
    #'  `itertools.product(m.enumerate_support())`.
    #'  @param expand (bool): whether to expand the support over the
    #'  batch dims to match the distribution's `batch_shape`.
    #'  @return Tensor iterating over dimension 0.
    enumerate_support = function(expand = TRUE) NULL,
    
    #'  Returns entropy of distribution, batched over batch_shape.
    #'  @return Tensor of shape batch_shape.
    entropy = function() NULL,
    
    #'  Returns perplexity of distribution, batched over batch_shape.
    #'  @param Tensor of shape batch_shape.
    perplexity = function() NULL,
    
    #' Returns the size of the sample returned by the distribution, given
    #' a `sample_shape`. Note, that the batch and event shapes of a distribution
    #' instance are fixed at the time of construction. If this is empty, the
    #' returned shape is upcast to (1,).
    #' @param sample_shape (torch.Size): the size of the sample to be drawn.
    .extended_shape = function(sample_shape=torch.Size()){
     if (!inherits(sample_shape, "torch_size"))
       sample_shape <- torch_size(sample_shape) #!
     sample_shape + self$batch_shape + self$event_shape
    },
    
    #' Argument validation for distribution methods such as `log_prob`,
    #' `cdf` and `icdf`. The rightmost dimensions of a value to be
    #' scored via these methods must agree with the distribution's batch
    #' and event shapes.
    #' @param     value (Tensor): the tensor whose log probability is to be
    #' computed by the `log_prob` method.
    .validate_sample = function(value){
      
    }
  

    def _validate_sample(self, value):
      """
    
        Args:
        
        Raises
            ValueError: when the rightmost dimensions of `value` do not match the
                distribution's batch and event shapes.
        """
    if not isinstance(value, torch.Tensor):
      raise ValueError('The value argument to log_prob must be a Tensor')
    
    event_dim_start = len(value.size()) - len(self._event_shape)
    if value.size()[event_dim_start:] != self._event_shape:
      raise ValueError('The right-most size of value must match event_shape: {} vs {}.'.
                       format(value.size(), self._event_shape))
    
    actual_shape = value.size()
    expected_shape = self._batch_shape + self._event_shape
    for i, j in zip(reversed(actual_shape), reversed(expected_shape)):
      if i != 1 and j != 1 and i != j:
      raise ValueError('Value is not broadcastable with batch_shape+event_shape: {} vs {}.'.
                       format(actual_shape, expected_shape))
    
    if not self.support.check(value).all():
      raise ValueError('The value argument must be within the support')
    
    def _get_checked_instance(self, cls, _instance=None):
      if _instance is None and type(self).__init__ != cls.__init__:
      raise NotImplementedError("Subclass {} of {} that defines a custom __init__ method "
                                "must also define a custom .expand() method.".
                                format(self.__class__.__name__, cls.__name__))
    return self.__new__(type(self)) if _instance is None else _instance
    
    def __repr__(self):
      param_names = [k for k, _ in self.arg_constraints.items() if k in self.__dict__]
    args_string = ', '.join(['{}: {}'.format(p, self.__dict__[p]
                                             if self.__dict__[p].numel() == 1
                                             else self.__dict__[p].size()) for p in param_names])
    return self.__class__.__name__ + '(' + args_string + ')'
    
    
    
  ),
  
  
  active = list(
    
    #' Returns a :class:`dist_constraint` object
    #' representing this distribution's support.
    support = function() NULL,
    
    #' Returns the mean on of the distribution
    mean = function() NULL,
    
    #' Returns the variance of the distribution
    variance = function() NULL,
    
    #' Returns the standard deviation of the distribution
    stddev = function() self$variance$sqrt()
  )
)