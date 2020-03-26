#' LU
#' 
#' Computes the LU factorization of a matrix or batches of matrices A. Returns a 
#' tuple containing the LU factorization and pivots of A. Pivoting is done if pivot 
#' is set to True.
#' 
#' @param A (Tensor) the tensor to factor of size (*, m, n)(∗,m,n)
#' @param pivot (bool, optional) – controls whether pivoting is done. Default: TRUE
#' @param get_infos (bool, optional) – if set to True, returns an info IntTensor. Default: FALSE
#' @param out (tuple, optional) – optional output tuple. If get_infos is True, then the elements 
#'   in the tuple are Tensor, IntTensor, and IntTensor. If get_infos is False, then the 
#'   elements in the tuple are Tensor, IntTensor. Default: NULL
#'   
#' @examples 
#' 
#' A = torch_randn(c(2, 3, 3))
#' torch_lu(A)
#' 
#' @export
torch_lu <- function(A, pivot=TRUE, get_infos=FALSE, out=NULL) {
  # If get_infos is True, then we don't need to check for errors and vice versa
  result <- torch__lu_with_info(A, pivot, get_infos)
  
  if (!is.null(out)) {
    if (!is.list(out))
      stop("argument 'out' must be a list of Tensors.")
    
    if (length(out) - as.integer(get_infos) != 2) {
      stop("expected tuple of ", 2 + as.integer(get_infos), " elements but got ",
           length(out))
    }
    
    for (i in seq_len(out)) {
      out[[i]] <- out[[i]]$resize_as_(result[[i]])$copy_(result[[i]])
    }
    
    return(out)
  }
  
 
  if (get_infos)
    return(result)
  else
    return(result[1:2])
}

torch_logical_not <- function(self) {
  .torch_logical_not(self)
}
