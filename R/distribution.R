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
    
    
    
    
  )
)