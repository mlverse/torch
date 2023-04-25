#' nn_utils_weight_norm
#' 
#' Applies weight normalization to a parameter in the given module.
#' 
#'         \eqn{\mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|}}
#'         
#' Weight normalization is a reparameterization that decouples the magnitude
#' of a weight tensor from its direction. This replaces the parameter specified
#' by `name`  (e.g. `'weight'`) with two parameters: one specifying the 
#' magnitude (e.g. `'weight_g'`) and one specifying the direction 
#' (e.g. `'weight_v'`).
#' 
#' @note
#' The pytorch Weight normalization is implemented via a hook that recomputes 
#' the weight tensor from the magnitude and direction before every `forward()` 
#' call. Since torch for R still do not support hooks, the weight recomputation
#' need to be done explicitly inside the `forward()` definition trough a call of
#' the `recompute()` method. See examples.
#'  
#' By default, with `dim = 0`, the norm is computed independently per output
#' channel/plane. To compute a norm over the entire weight tensor, use
#' `dim = NULL`.
#'  
#'  @references https://arxiv.org/abs/1602.07868
#'  
#' @param module (Module): containing module
#' @param name (str, optional): name of weight parameter
#' @param dim (int, optional): dimension over which to compute the norm
#' 
#' @returns The original module with the weight_v and weight_g paramters.
#' 
#' @examples
#' x = nn_linear(in_features = 20, out_features = 40)
#' weight_norm = nn_utils_weight_norm$new(name = 'weight', dim = 2)
#' weight_norm$apply(x)
#' x$weight_g$size()
#' x$weight_v$size()
#' x$weight
#' 
#' # the recompute() method recomputes the weight using g and v. It must be called
#' explicitly inside `forward()`.
#' weight_norm$recompute(x)
#' 
#' @export
nn_utils_weight_norm <- R6::R6Class(
  "WeightNorm",
  lock_objects = FALSE,
  public = list(
    name = NULL,
    dim = NULL,
    initialize =  function(name, dim) {
      if (is.null(dim)) 
        dim = -1
      
      self$name <- name
      self$dim <- dim
    },
    
    compute_weight = function(module, name = NULL, dim = NULL) { 
      name = name %||% self$name
      dim = dim %||% self$dim %||% -1
      g = module[[paste0(name, '_g')]]
      v = module[[paste0(name, '_v')]]
      return(torch__weight_norm(v, g, dim))
    },
    
    apply = function(module, name = NULL, dim = NULL) {
      name = name %||% self$name
      dim = dim %||% self$dim
      weight = module[[name]] 
      
      # remove w from parameter list
      module$register_parameter(name, NULL)
      
      # add g and v as new parameters and express w as g/||v|| * v
      module$register_parameter(paste0(name, '_g'), nn_parameter(torch_norm_except_dim(weight, 2, dim)$data()))
      module$register_parameter(paste0(name, '_v'), nn_parameter(weight$data()))
      module[[name]] <- self$compute_weight(module)
      
      return(invisible(module))
    },
    
    call = function(module) { 
      module[[self$name]] <- self$compute_weight(module)
      return(invisible(module))
    },
    
    recompute = function(module) {
      self$call(module)
    },
    
    remove = function(module, name = NULL) {
      name = name %||% self$name
      weight <- self$compute_weight(module)
      module$register_parameter(paste0(name, '_g'), NULL)
      module$register_parameter(paste0(name, '_v'), NULL)
      module[[name]] <- nn_parameter(weight$data())
    }
  )
)


