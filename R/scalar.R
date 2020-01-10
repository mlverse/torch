Scalar <- R6::R6Class(
  classname = "torch_scalar", 
  
  public = list(
    
    ptr = NULL,
    
    initialize = function(x, ptr = NULL) {
      
      if (!is.null(ptr)) {
        self$ptr <- ptr
        return(NULL)
      }
      
      self$ptr <- cpp_torch_scalar(x);
      
    }
    
  )
  
)

torch_scalar <- function(x) {
  Scalar$new(x)
}

