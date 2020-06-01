nnf_adaptive_avg_pool1d <- function(input, output_size) {
  torch_adaptive_avg_pool1d(input, output_size)
}

nnf_adaptive_avg_pool2d <- function(input, output_size) {
  torch_adaptive_avg_pool2d(input, output_size)
}

nnf_adaptive_avg_pool3d <- function(input, output_size) {
  torch_adaptive_avg_pool3d(input, output_size)
}

nnf_adaptive_max_pool1d <- function(input, output_size, return_indices = FALSE) {
  o <- torch_adaptive_max_pool1d(input, output_size)
  if (!return_indices)
    o <- o[[1]]
  
  o
}

nnf_adaptive_max_pool1d_with_indices <- function(input, output_size) {
  torch_adaptive_max_pool1d(input, output_size)
}

nnf_adaptive_max_pool2d <- function(input, output_size, return_indices = FALSE) {
  o <- torch_adaptive_max_pool2d(input, output_size)
  if (!return_indices)
    o <- o[[1]]
  
  o
}

nnf_adaptive_max_pool2d_with_indices <- function(input, output_size) {
  torch_adaptive_max_pool2d(input, output_size)
}

nnf_adaptive_max_pool3d <- function(input, output_size, return_indices = FALSE) {
  o <- torch_adaptive_max_pool3d(input, output_size)
  if (!return_indices)
    o <- o[[1]]
  
  o
}

nnf_adaptive_max_pool3d_with_indices <- function(input, output_size) {
  torch_adaptive_max_pool3d(input, output_size)
}

nnf_affine_grid <- function(theta, size, align_corners = FALSE) {
  torch_affine_grid_generator(theta, size, align_corners)
}

nnf_avg_pool1d <- function(input, kernel_size, stride = NULL, padding = 0, ceil_mode = FALSE, 
                           count_include_pad = TRUE) {
  torch_avg_pool1d(input, kernel_size, stride, padding, ceil_mode, count_include_pad)
}

nnf_avg_pool2d <- function(input, kernel_size, stride = NULL, padding = 0, ceil_mode = FALSE, 
                           count_include_pad = TRUE, divisor_override = NULL) {
  torch_avg_pool2d(input, kernel_size, stride, padding, ceil_mode, count_include_pad,
                   divisor_override)
}

nnf_avg_pool3d <- function(input, kernel_size, stride = NULL, padding = 0, ceil_mode = FALSE, 
                           count_include_pad = TRUE, divisor_override = NULL) {
  torch_avg_pool3d(input, kernel_size, stride, padding, ceil_mode, count_include_pad,
                   divisor_override)
}

nnf_bilinear <- function(input1, input2, weight, bias = NULL) {
  torch_bilinear(input1, input2, weight, bias)
}









nnf_conv_tbc <- function(input, weight, bias, pad = 0) {
  torch_conv_tbc(self = input, weight = weight, bias = bias, pad = pad)
}













nnf_fractional_max_pool2d <- function(input, kernel_size, output_size=NULL,
                                      output_ratio=NULL, return_indices=FALSE,
                                      random_samples = NULL) {
  
  if (is.null(output_size)) {
    output_ratio_ <- pair(output_ratio)
    output_size <- list(
      as.integer(input$size(2) * output_ratio_[[1]]),
      as.integer(input$size(3) * output_ratio_[[2]])
    )
  }
  
  if (is.null(random_samples)) {
    random_samples <- torch_rand(input$size(0), input$size(1), 2, 
                                 dtype = input$dtype, device = input$device)
  }
  
  res <- torch_fractional_max_pool2d(self = input, kernel_size = kernel_size, 
                              output_size = output_size, random_samples = random_samples)
  
  if (return_indices)
    res
  else
    res[[1]]
}


nnf_fractional_max_pool3d <- function(kernel_size, output_size = NULL, output_ratio = NULL, 
                                      return_indices = FALSE, random_samples = NULL) {
  
  
  if (is.null(output_size)) {
    
    if (length(output_size) == 1)
      
      output_ratio_ <- rep(output_size, 3)
      output_size <- c(
        input$size(2) * output_ratio_[1],
        input$size(3) * output_ratio_[2],
        input$size(4) * output_ratio_[3],
      )
  }
  
  if (is.null(random_samples)) {
    random_samples <- torch_rand(input$size(0), input$size(1), 3, dtype = input$dtype, 
                                 device = input$device)
  }
  
  res <- torch_fractional_max_pool3d(
    self = input, 
    kernel_size = kernel_size,
    output_size = output_size,
    random_samples = random_samples
  )
  
  
  if (return_indices)
    res
  else
    res[[1]]
}

nnf_grid_sample <- function(input, grid, mode = c("bilinear", "nearest"), 
                            padding_mode = c("zeros", "border", "reflection"), 
                            align_corners = FALSE) {
  
  if (mode == "bilinear")
    mode_enum <- 0
  else if (mode == "nearest")
    mode_enum <- 1
  else
    value_error("Unknown mode name '{mode}'. Supported modes are 'bilinear'",
                "and 'nearest'.")
  
  
  if (padding_mode == "zeros")
    padding_mode_enum <- 0
  else if (padding_mode == "border")
    padding_mode_enum <- 1
  else if (padding_mode == "reflection")
    padding_mode_enum <- 2
  else
    value_error("Unknown padding mode name '{padding_mode}'. Supported modes are",
                "'zeros', 'border' and 'reflection'.")
  
  torch_grid_sampler(input = input, grid = grid, interpolation_mode = mode_enum,
                     padding_mode = padding_mode_enum, align_corners = align_corners)
}



nnf_hardsigmoid <- function(input, inplace = FALSE) {
  if (inplace)
    torch_hardsigmoid_(input)
  else
    torch_hardsigmoid(input)
}


interp_output_size <- function(input, size, scale_factor, recompute_scale_factor) {
  dim <- input$dim() - 2
  
  if (is.null(size) && is.null(scale_factor))
    value_error("either size or scale_factor should be defined")
  
  if (!is.null(size) && !is.null(scale_factor))
    value_error("only one of size or scale_factor should be defined")
  
  if (!is.null(scale_factor)) {
    if (length(scale_factor) != dim)
      value_error("scale_factor shape must match input shape.", 
                  "Input is {dim}D, scale_factor size is {lenght(scale_factor)}")
    
  }
  
  if (!is.null(size)) {
    if (length(size) > 1)
      return(size)
    else
      return(rep(size, dim))
  }
  
  if (is.null(scale_factor))
    value_error("assertion failed")
  
  if (length(scale_factor) > 1)
    scale_factors <- scale_factor
  else
    scale_factors <- rep(scale_factor, dim)
  
  
  if (is.null(recompute_scale_factor)) {
    is_float_scale_factor <- FALSE
    
    for (scale in scale_factors) {
      is_float_scale_factor <- floor(scale) != scale
      if (is_float_scale_factor)
        break
    }
    
    warn("The default behavior for interpolate/upsample with float scale_factor will change ",
         "in 1.6.0 to align with other frameworks/libraries, and use scale_factor directly, ",
         "instead of relying on the computed output size. ",
         "If you wish to keep the old behavior, please set recompute_scale_factor=True. ",
         "See the documentation of nn.Upsample for details. ")
  }
  
  
  lapply(
    seq_len(dim),
    function(i) {
      i <- i - 1
      floor(input$size(i + 2) * scale_factors[i + 1])
    }
  )
}

nnf_interpolate <- function(input, size = NULL, scale_factor = NULL, 
                            mode = "nearest", align_corners = FALSE, 
                            recompute_scale_factor = NULL) {
  
  scale_factor_len <- input$dim() - 2
  if (length(scale_factor) == 1)
    scale_factor_repeated <- rep(scale_factor, scale_factor_len)
  else
    scale_factor_repeated <- scale_factor
  
  sfl <- scale_factor_repeated
  sze <- interp_output_size(input, size = size, scale_factor = scale_factor, 
                                    recompute_scale_factor = recompute_scale_factor)
  
  if (input$dim() == 3 && mode == "nearest") {
    return(torch_upsample_nearest1d(input, output_size = sze, scales = sfl[[1]]))
  }
  
  if (input$dim() == 4 && mode == "nearest") {
    return(torch_upsample_nearest2d(input, output_size = sze, scales_h = sfl[[1]],
                                    scales_w = sfl[[2]]))
  }
  
  if (input$dim() == 5 && mode == "nearest") {
    return(torch_upsample_nearest3d(input, sze, sfl[[1]], sfl[[2]], sfl[[3]]))
  }
  
  if (input$dim() == 3 && mode == "area") {
    return(torch_adaptive_avg_pool1d(input, sze))
  }
  
  if (input$dim() == 4 && mode == "area") {
    return(torch_adaptive_avg_pool2d(input, sze))
  }
  
  if (input$dim() == 5 && mode == "area") {
    return(torch_adaptive_avg_pool3d(input, sze))
  }
  
  if (input$dim() == 3 && mode == "linear") {
    return(torch_upsample_linear1d(input, sze, align_corners, sfl[[1]]))
  }
  
  if (input$dim() == 3 && mode == "bilinear") {
    not_implemented_error("Got 3D input, but bilinear mode needs 4D input")
  }
  
  if (input$dim() == 3 && mode == "trilinear") {
    not_implemented_error("Got 3D input, but trilinear mode needs 5D input")
  }
  
  if (input$dim() == 4 && mode == "linear") {
    not_implemented_error("Got 4D input, but trilinear mode needs 3D input")
  }
  
  if (input$dim() == 4 && mode == "bilinear") {
    return(torch_upsample_bilinear2d(input, sze, align_corners, sfl[[1]], sfl[[2]]))
  }
  
  if (input$dim() == 4 && mode == "trilinear") {
    not_implemented_error("Got 4D input, but trilinear mode needs 5D input")
  }
  
  if (input$dim() == 5 && mode == "linear") {
    not_implemented_error("Got 5D input, but trilinear mode needs 3D input")
  }
    
  if (input$dim() == 5 && mode == "bilinear") {
    not_implemented_error("Got 5D input, but bilinear mode needs 4D input")
  }
  
  if (inpt$dim() == 5 && mode == "trilinear") {
    return(torch_upsample_trilinear3d(input, sze, align_corners, sfl[[1]], sfl[[2]], 
                               sfl[[3]]))
  }
  
  if (input$dim() ==4 && mode == "bicubic") {
    return(torch_upsample_bicubic2d(input, sze, align_corners, sfl[[1]], sfl[[2]]))
  }
  
  not_implemented_error("Input Error: Only 3D, 4D and 5D input Tensors supported",
                        " (got {input$dim()}D) for the modes: nearest | linear | bilinear | bicubic | trilinear",
                        " (got {mode})")
}








nnf_lp_pool1d <- function(input, norm_type, kernel_size, stride = NULL, 
                          ceil_mode = FALSE) {
  
  if (!is.null(stride)) {
    out <- nnf_avg_pool1d(input$pow(norm_type), kernel_size, stride, 0, ceil_mode)
  } else {
    out <- nnf_avg_pool1d(input$pow(norm_type), kernel_size, padding = 0, 
                          ceil_mode = ceil_mode)
    
  }
  
  (torch_sign(out) * nnf_relu(torch_abs(out)))$mul(kernel_size)$pow(1/norm_type)
}

nnf_lp_pool2d <- function(input, norm_type, kernel_size, stride = NULL, 
                          ceil_mode = FALSE) {
  
  k <- nn_util_pair(kernel_size)
  if (!is.null(stride)) {
    out <- nnf_avg_pool2d(input$pow(norm_type), kernel_size, stride, 0, ceil_mode)
  } else {
    out <- nnf_avg_pool2d(input$pow(norm_type), kernel_size, padding = 0, 
                          ceil_mode = ceil_mode)
  }
  
  (torch$sign(out) * nnf_relu(torch$abs(out)))$mul(k[[1]] * k[[2]])$pow(1/norm_type)
}

nnf_max_pool1d <- function(input, kernel_size, stride=NULL, padding=0, dilation=1,
                           ceil_mode=FALSE, return_indices=FALSE) {
  
  if (return_indices)
    torch_max_pool1d_with_indices(input, kernel_size, stride, padding, dilation,
                     ceil_mode)
  else
    torch_max_pool1d(input, kernel_size, stride, padding, dilation,
                     ceil_mode)
}


nnf_max_pool2d <- function(input, kernel_size, stride=NULL, padding=0, dilation=1,
                           ceil_mode=FALSE, return_indices=FALSE) {
  if (return_indices)
    torch_max_pool2d_with_indices(input, kernel_size, stride, padding, dilation,
                                  ceil_mode)
  else
    torch_max_pool2d(input, kernel_size, stride, padding, dilation,
                     ceil_mode)
}

nnf_max_pool3d <- function(input, kernel_size, stride=NULL, padding=0, dilation=1,
                           ceil_mode=FALSE, return_indices=FALSE) {
  if (return_indices)
    torch_max_pool3d_with_indices(input, kernel_size, stride, padding, dilation,
                                  ceil_mode)
  else
    torch_max_pool3d(input, kernel_size, stride, padding, dilation,
                     ceil_mode)
}


unpool_output_size <- function(input, kernel_size, stride, padding, output_size) {
  
  input_size <- input$size()
  default_size <- list()
  for (d in seq_along(kernel_size)) {
    default_size[[d]] <- (input_size[d+2] - 1) * stride[d] + kernel_size[d] - 
      2 * padding[d]
  }
  
  if (is.null(output_size)) {
    ret <- default_size
  } else {
    
    if (length(output_size) == (length(kernel_size) + 2)) {
      output_size <- output_size[-c(1,2)] 
    }
    
    if (length(output_size) != length(kernel_size)) {
      value_error("output_size should be a sequence containing ",
                  "{length(kernel_size)} or {length(kernel_size) + 2} elements", 
                  "but it has a length of '{length(output_size)}'")
    }
    
    for (d in seq_along(kernel_size)) {
      min_size <- default_size[d] - stride[d]
      max_size <- default_size[d] + stride[d]
    }
    
    ret <- output_size
    
  }
    
  ret
}

nnf_max_unpool1d <- function(input, indices, kernel_size, stride = NULL,
                             padding = 0, output_size = NULL) {
  if (is.null(stride))
    stride <- kernel_size
  
  output_size <- unpool_output_size(input, kernel_size, stride, padding,
                                    output_size)
  
  output_size <- c(output_size, 1)
  
  torch_max_unpool2d(input$unsqueeze(3), indices$unsqueeze(3), output_size)$squeeze(3)
}

nnf_max_unpool2d <- function(input, indices, kernel_size, stride = NULL,
                             padding = 0, output_size = NULL) {
  
  kernel_size <- nn_util_pair(kernel_size)
  if(is.null(stride))
    stride <- kernel_size
  else
    stride <- nn_util_pair(stride)
  
  padding <- nn_util_pair(padding)
  
  output_size <- unpool_output_size(input, kernel_size, stride, padding,
                                    output_size)
  
  torch_max_unpool2d(input, indices, output_size)
}

nnf_max_unpool3d <- function(input, indices, kernel_size, stride = NULL,
                             padding = 0, output_size = NULL){
  
  kernel_size <- nn_util_triple(kernel_size)
  padding <- nn_util_triple(padding)
  if (is.null(stride))
    stride <- kernel_size
  else
    stride <- nn_util_triple(stride)
  
  output_size <- unpool_output_size(input, kernel_size, stride, padding,
                                    output_size)
  
  torch_max_unpool3d(input, indices, output_size, stride, padding)
}



nnf_one_hot <- function(tensor, num_classes = -1) {
  torch_one_hot(tensor, num_classes)
}













