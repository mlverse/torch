#' A simplified version of torchvision.utils.make_grid.
#' 
#' Arranges a batch of (image) tensors in a grid, with optional padding between images.
#' Expects a 4d mini-batch tensor of shape (B x C x H x W).
#' 
#' @param tensor tensor to arrange in grid
#' @param scale whether to normalize (min-max-scale) the input tensor
#' @param num_rows number of rows making up the grid (default 8)
#' @param padding amount of padding between batch images (default 2)
#' @param pad_value pixel value to use for padding
#'
#' @export
vision_make_grid <-
  function(tensor,
           scale = TRUE,
           num_rows = 8,
           padding = 2,
           pad_value = 0) {
    
    min_max_scale <- function(x) {
      min = x$min()$item()
      max = x$max()$item()
      x$clamp_(min = min, max = max)
      x$add_(-min)$div_(max - min + 1e-5)
      x
    }
    if(scale) tensor <- min_max_scale(tensor)
    
    nmaps <- tensor$size(1)
    xmaps <- min(num_rows, nmaps)
    ymaps <- ceiling(nmaps / xmaps)
    height <- floor(tensor$size(3) + padding)
    width <- floor(tensor$size(4) + padding)
    num_channels <- tensor$size(2)
    grid <-
      tensor$new_full(c(num_channels, height * ymaps + padding, width * xmaps + padding),
                      pad_value)
    k <- 0
    
    for (y in 0:(ymaps - 1)) {
      for (x in 0:(xmaps - 1)) {
        if (k >= nmaps)
          break
        grid$narrow(
          dim = 2,
          start = torch_tensor(y * height + padding, dtype = torch_int64())$sum(dim = 1),
          length = height - padding
        )$narrow(
          dim = 3,
          start = torch_tensor(x * width + padding, dtype = torch_int64())$sum(dim = 1),
          length = width - padding
        )$copy_(tensor[k + 1, , ,])
        k <- k + 1
      }
    }
    
    grid
  }