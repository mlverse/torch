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

nnf_alpha_dropout <- function(input, p = 0.5, training = FALSE, inplace = FALSE) {
  if (inplace)
    torch_alpha_dropout_(input, p, training)
  else
    torch_alpha_dropout(input, p, training)
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

nnf_batch_norm <- function(input, running_mean, running_var, weight = NULL, bias = NULL,
                           training = FALSE, momentum = 0.1, eps = 1e-5) {
  torch_batch_norm(input = input, weight = weight, bias = bias, running_mean = running_mean,
                   running_var = running_var, training = training, momentum = momentum,
                   eps = eps, cudnn_enabled = backends_cudnn_enabled())
}

nnf_bilinear <- function(input1, input2, weight, bias = NULL) {
  torch_bilinear(input1, input2, weight, bias)
}

nnf_binary_cross_entropy <- function(input, target, weight = NULL, size_average = NULL, 
                                     reduction = c("mean", "sum", "none")) {
  torch_binary_cross_entropy(input, target, weight, size_average=size_average,
                             reduction=reduction_enum(reduction))
}

nnf_binary_cross_entropy_with_logits <- function(input, target, weight = NULL, 
                                                 size_average = NULL, 
                                                 reduction = c("mean", "sum", "none"), 
                                                 pos_weight = NULL) {
  torch_binary_cross_entropy_with_logits(input, target, weigth, pos_weight, 
                                         reduction_enum(reduction))
}

nnf_celu <- function(input, alpha = 1, inplace = FALSE) {
  if (inplace)
    torch_celu_(input, alpha)
  else
    torch_celu(input, alpha)
}

nnf_celu_ <- function(input, alpha = 1) {
  torch_celu_(input, alpha)
}

nnf_conv1d <- function(input, weight, bias = NULL, stride = 1, padding = 0, dilation = 1, 
                       groups = 1) {
  torch_conv1d(
    input = input, weight = weight, bias = bias, stride = stride,
    padding = padding, dilation = dilation, groups = groups
  )
}

nnf_conv2d <- function(input, weight, bias = NULL, stride = 1, padding = 0, dilation = 1, 
                       groups = 1) {
  torch_conv2d(
    input = input, weight = weight, bias = bias, stride = stride,
    padding = padding, dilation = dilation, groups = groups
  )
}

nnf_conv3d <- function(input, weight, bias = NULL, stride = 1, padding = 0, dilation = 1, 
                       groups = 1) {
  torch_conv3d(
    input = input, weight = weight, bias = bias, stride = stride,
    padding = padding, dilation = dilation, groups = groups
  )
}

nnf_conv_tbc <- function(input, weight, bias, pad = 0) {
  torch_conv_tbc(self = input, weight = weight, bias = bias, pad = pad)
}

nnf_conv_transpose1d <- function(input, weight, bias=NULL, stride=1, padding=0, 
                                 output_padding=0, groups=1, dilation=1) {
  torch_conv_transpose1d(
    input = input, weight = weight, bias = bias, stride = stride,
    padding = padding, output_padding = output_padding, groups = groups,
    dilation = dilation
  )
}

nnf_conv_transpose2d <- function(input, weight, bias=NULL, stride=1, padding=0, 
                                 output_padding=0, groups=1, dilation=1) {
  torch_conv_transpose2d(
    input = input, weight = weight, bias = bias, stride = stride,
    padding = padding, output_padding = output_padding, groups = groups,
    dilation = dilation
  )
}

nnf_conv_transpose3d <- function(input, weight, bias=NULL, stride=1, padding=0, 
                                 output_padding=0, groups=1, dilation=1) {
  torch_conv_transpose3d(
    input = input, weight = weight, bias = bias, stride = stride,
    padding = padding, output_padding = output_padding, groups = groups,
    dilation = dilation
  )
}

nnf_cosine_embedding_loss <- function(input1, input2, target, margin=0, 
                                      size_average=NULL, reduction=c("mean", "sum", "none")) {
  torch_cosine_embedding_loss(input1 = input1, input2 = input2, target = target, 
                              margin = margin, reduction = reduction_enum(reduction))
}

nnf_cosine_similarity <- function(x1, x2, dim=1, eps=1e-8) {
  torch_cosine_similarity(x1 = x1, x2 = x2, dim = dim, eps = eps)
}

nnf_cross_entropy <- function(input, target, weight=NULL, ignore_index=-100, 
                              reduction=c("mean", "sum", "none")) {
  torch_nll_loss(self = torch_log_softmax(input, 1), target = target, weight = weight, 
                 reduction = reduction_enum(reduction), ignore_index = ignore_index)
}

nnf_ctc_loss <- function(log_probs, targets, input_lengths, target_lengths, blank=0,
                         reduction=c('mean', "sum", "none"), zero_infinity=FALSE) {
  torch_ctc_loss(log_probs = log_probs, targets = targets, input_lengths = input_lengths,
                 target_lengths = target_lengths, blank = blank, reduction = reduction_enum(reduction),
                 zero_infinity = zero_infinity)
}

nnf_dropout <- function(input, p=0.5, training=TRUE, inplace=FALSE) {
  if (inplace)
    torch_dropout_(input, p, training)
  else
    torch_dropout(input, p, training)
}

nnf_dropout2d <- function(input, p=0.5, training=TRUE, inplace=FALSE) {
  if (inplace)
    torch_feature_dropout_(input, p, training)
  else
    torch_feature_dropout(input, p, training)
}

nnf_dropout3d <- function(input, p=0.5, training=TRUE, inplace=FALSE) {
  if (inplace)
    torch_feature_dropout_(input, p, training)
  else
    torch_feature_dropout(input, p, training)
}

nnf_elu <- function(input, alpha=1, inplace=FALSE) {
  if(inplace)
    torch_elu_(input, alpha = alpha)
  else
    torch_elu(input, alpha = alpha)
}

nnf_elu_ <- function(input, alpha=1) {
  torch_elu_(input, alpha = alpha)
}

nnf_embedding <- function(input, weight, padding_idx=NULL, max_norm=NULL, norm_type=2,
                          scale_grad_by_freq=FALSE, sparse=FALSE) {
  if (is.null(padding_idx))
    padding_idx <- -1
  
  if (!is.null(max_norm)) {
    input <- input$contiguous()
    with_no_grad({
      torch_embedding_renorm_(weight, input, max_norm, norm_type)
    })
  }
  
  torch_embedding(weight = weight, input = input, padding_idx = padding_idx,
                  scale_grad_by_freq = scale_grad_by_freq, sparse = sparse)  
}

nnf_embedding_bag <- function(input, weight, offsets = NULL, max_norm = NULL, 
                              norm_type = 2, scale_grad_by_freq = FALSE, 
                              mode = "mean", sparse= FALSE, per_sample_weights = NULL,
                              include_last_offset = FALSE) {

  if (input$dim() == 2) {
    input <- input$reshape(-1)
    if (!is.null(per_sample_weights)) {
      per_sample_weights <- per_sample_weights$reshape(-1)
    }
  } 
  
  if (mode == 'sum') {
    mode_enum <- 0
  } else if (mode == "mean") {
    mode_enum <- 1
  } else if (mode == "max") {
    mode_enum <- 2
  }
    
  if (!is.null(max_norm)) {
    input <- input$contiguous()
    with_no_grad({
      torch_embedding_renorm_(weight, input, max_norm, norm_type)
    })
  }
  
  ret <- torch_embedding_bag(weight = weight, indices = input, offsets = offsets, 
                      scale_grad_by_freq = scale_grad_by_freq, mode = mode_enum,
                      sparse = sparse, per_sample_weights = per_sample_weights, 
                      include_last_offset = include_last_offset)
                      
  ret[[1]]
}

pair <- function(x) {
  if (length(x) == 1)
    rep(x, 2)
  else
    x
}

nnf_fold <- function(input, output_size, kernel_size, dilation=1, padding=0, stride=1) {
  torch_col2im(self = input, output_size = pair(output_size), 
               kernel_size = pair(kernel_size), dilation = pair(dilation), 
               padding = pair(padding), stride = pair(stride))
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

nnf_gelu <- function(input) {
  torch_gelu(self = input)
}

nnf_glu <- function(input, dim = -1) {
  torch_glu(self = input, dim = dim)
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

nnf_group_norm <- function(input, num_groups, weight = NULL, bias = NULL,
                           eps = 1e-5) {
  
  torch_group_norm(input, input, num_groups = num_groups, weight = weight,
                   bias = bias, eps = eps #TODO ,cudnn_enabled = backends_cudnn_enabled
  )
}

nnf_gumbel_softmax <- function(logits, tau = 1, hard = FALSE, dim = -1) {
  gumbels <- -torch_empty_like(logits, memory_format = torch_contiguous_format())
  gumbels <- gumbels$exponential_()$log()
  gumbels <- (logits + gumbels) / tau
  y_soft <- gumbels$softmax(dim)
  
  if (hard) {
    index <- y_soft$max(dim, keepdim = TRUE)[[2]]
    y_hard <- torch_zeros_like(logits, memory_format = torch_contiguous_format())
    y_hard <- y_hard$scatter_(dim, index, 1)
    ret <- y_hard - y_soft$detach() + y_soft
  } else {
    ret = y_soft
  }
  ret
}

nnf_hardshrink <- function(input, lambd = 0.5) {
  torch_hardshrink(input, lambd)
}

nnf_hardsigmoid <- function(input, inplace = FALSE) {
  if (inplace)
    torch_hardsigmoid_(input)
  else
    torch_hardsigmoid(input)
}

nnf_hardtanh <- function(input, min_val = -1, max_val = 1, inplace = FALSE) {
  if (inplace)
    torch_hardtanh_(input, min_val, max_val)
  else
    torch_hardtanh(input, min_val, max_val)
}

nnf_hardtanh_ <- function(input, min_val = -1, max_val = 1) {
  nnf_hardtanh(input, min_val, max_val, inplace = TRUE)
}

nnf_hinge_embedding_loss <- function(input, target, margin = 1, reduction = "mean") {
  torch_hinge_embedding_loss(input, target, margin, 
                             reduction = reduction_enum(reduction))
}

nnf_instance_norm <- function(input, running_mean = NULL, running_var = NULL,
                              weight = NULL, bias = NULL, use_input_stats = TRUE,
                              momentum = 0.1, eps = 1e-5) {
  
  torch_instance_norm(input, weight, bias, running_mean, running_var,
                      use_input_stats, momentum, eps, FALSE #TODO backend_cudnn_enabled)
  )
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

nnf_kl_div <- function(input, target, reduction = "mean") {
  
  if (reduction == "mean")
    warn("reduction: 'mean' divides the total loss by both the batch size and the support size.",
         "'batchmean' divides only by the batch size, and aligns with the KL div math definition.",
         "'mean' will be changed to behave the same as 'batchmean' in the next major release.")

  
  if (reduction == "batchmean")
    red_enum <- reduction_enum("sum")
  else
    red_enum <- reduction_enum(reduction)
  
  reduced <- torch_kl_div(input, target, red_enum)
  
  if (reduction == "batchmean")
    reduced <- reduced/ input$size(0)
  
  reduced
}

nnf_l1_loss <- function(input, target, reduction = "mean") {
  
  if (target$requires_grad) {
    ret <- torch_abs(input - target)
    if (!is.null(reduction)) {
      
      if (reduction == "mean")
        ret <- torch_mean(ret)
      else
        ret <- torch_sum(ret)
      
    }
  } else {
    expanded <- torch_broadcast_tensors(input, target)
    ret <- torch_l1_loss(expanded[[1]], expanded[[2]], reduction_enum(reduction))
  }
  
  ret
}

nnf_layer_norm <- function(input, normalized_shape, weight = NULL, bias = NULL,
                           eps = 1e-5) {
  torch_layer_norm(
    input, normalized_shape, weight, bias, eps, FALSE #TODO backends_cudnn_enabled
  )
}

nnf_leaky_relu <- function(input, negative_slope = 0.01, inplace = FALSE) {
  if (inplace)
    torch_leaky_relu_(input, negative_slope)
  else
    torch_leaky_relu(input, negative_slope)
}


nnf_linear <- function(input, weight, bias = NULL) {
  
  if (input$dim() == 2 && !is.null(bias)) {
    ret <- torch_addmm(bias, input, weight$t())
  } else {
    output <- input$matmul(weight$t())
    if (!is.null(bias))
      output <- output + bias
    ret <- output
  }
  
  ret
}

nnf_local_response_norm <- function(input, size, alpha = 1e-4, beta = 0.75, k = 1) {
  
  dim <- input$dim()
  div <- input$mul(input)$unsqueeze(1)
  
  if (dim == 3) {
    div <- nnf_pad(div, c(0, 0, as.integer(size/2), as.integer((size - 1)/2)))
    div <- nnf_avg_pool2d(div, c(size, 1), stride = 1)$squeeze(1)
  } else {
    sizes <- input$size()
    div <- div$view(sizes[1], 1, sizes[2], sizes[3], -1)
    div <- nnf_pad(div, c(0,0,0,0, as.integer(size/2), as.integer((size - 1)/2)))
    div <- nnf_avg_pool3d(div, c(size, 1, 1), stride = 1)$squeeze(1)
    div <- div$view(sizes)
  }
  
  div <- div$mul(alpha)$add(k)$pow(beta)
  input/div
}

nnf_log_softmax <- function(input, dim = NULL, dtype = NULL, ...) {

  if (is.null(dtype))
    ret <- input$log_softmax(dim())
  else
    ret <- input$log_softmax(dim, dtype = dtype)
  
  ret  
}

nnf_logsigmoid <- function(input) {
  torch_log_sigmoid(input)
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

nnf_margin_ranking_loss <- function(input1, input2, target, margin = 0,
                                    reduction = "mean") {
  torch_margin_ranking_loss(input1, input2, target, margin, 
                            reduction_enum(reduction))
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

nnf_mse_loss <- function(input, target, reduction = "mean") {
  
  if (target$requires_grad) {
    ret <- (input - target) ^ 2
    if (!is.null(reduction)) {
      
      if (reduction == "mean")
        ret <- torch_mean(ret)
      else
        ret <- torch_sum(ret)
      
    }
  } else {
    
    expanded <- torch_broadcast_tensors(list(input, target))
    ret <- torch_mse_loss(expanded[[1]], expanded[[2]], reduction_enum(reduction))
    
  }
  
  ret
}

nnf_multi_head_attention_forward <- function(
  query,                           # type: Tensor
  key,                             # type: Tensor
  value,                           # type: Tensor
  embed_dim_to_check,              # type: int
  num_heads,                       # type: int
  in_proj_weight,                  # type: Tensor
  in_proj_bias,                    # type: Tensor
  bias_k,                          # type: Optional[Tensor]
  bias_v,                          # type: Optional[Tensor]
  add_zero_attn,                   # type: bool
  dropout_p,                       # type: float
  out_proj_weight,                 # type: Tensor
  out_proj_bias,                   # type: Tensor
  training=TRUE,                   # type: bool
  key_padding_mask=NULL,           # type: Optional[Tensor]
  need_weights=TRUE,               # type: bool
  attn_mask=NULL,                  # type: Optional[Tensor]
  use_separate_proj_weight=FALSE,  # type: bool
  q_proj_weight=NULL,              # type: Optional[Tensor]
  k_proj_weight=NULL,              # type: Optional[Tensor]
  v_proj_weight=NULL,              # type: Optional[Tensor]
  static_k=NULL,                   # type: Optional[Tensor]
  static_v=NULL                    # type: Optional[Tensor])
  ) {

  o <- query$size()
  tgt_len <- o[[1]]; bsz <- o[[2]]; embed_dim <- o[[3]];
  
  
  head_dim <- floor(embed_dim / num_heads) 
  
  scaling <- head_dim^(-0.5)
  
  if (!use_separate_proj_weight) {
    
    if (torch_equal(query, key) & torch_equal(key, value)) {
      # self-attention
      o <- nnf_linear(query, in_proj_weight, in_proj_bias)$chunk(3, dim = -1)
      q <- o[[1]]; k <- o[[2]]; v <- o[[3]]
    } else if (torch_equal(key, value)) {
      # encoder-decoder attention
      #             # This is inline in_proj function with in_proj_weight and in_proj_bias
      b_ <- in_proj_bias
      start_ <- 1
      end_ <- embed_dim
      w_ <- in_proj_weight[start_:end_,]
      
      if (!is.null(b_)) {
        b_ <- b_[start_:end_] 
      }
      
      q <- nnf_linear(query, w_, b_)
      
      if (is.null(key)) {
        k <- NULL
        v <- NULL
      } else {
        b_ <- in_proj_bias
        start_ <- embed_dim
        end_ <- NULL
        w_ <- in_proj_weight[start_:N, ]
        if (!is.null(b_)) {
          b_ <- b_[start_:N]
          o <- nnf_linear(key, w_, b_)$chunk(2, dim = -1)
          k <- o[[1]]; v <- o[[2]]
        }
        
      }
      
    } else {
      
      # This is inline in_proj function with in_proj_weight and in_proj_bias
      b_ <- in_proj_bias
      start_ <- 0
      end_ <- embed_dim
      w_ <- in_proj_weight[start_:end_, ]
      if (!is.null(b_))
        b_ <- b_[start_:end_]
      q <- nnf_linear(query, w_, b_)
      
      
      # This is inline in_proj function with in_proj_weight and in_proj_bias
      b_ <- in_proj_bias
      start_ <- embed_dim
      end_ <- embed_dim * 2
      w_ <- in_proj_weight[start_:end_,]
      if (!is.null(b_))
        b_ <- b_[start_:end_]
      k <- nnf_linear(key, w_, b_)
      
      # This is inline in_proj function with in_proj_weight and in_proj_bias
      b_ <- in_proj_bias
      start_ <- embed_dim * 2
      end_ <- NULL
      w_ <- in_proj_weight[start_:N,]
      if (!is.null(b_)) {
        b_ <- b_[start_:N]
      }
      v <- nnf_linear(value, w_, b_)
      
    }
    
  } else {
    
    if (!is.null(in_proj_bias)) {
      q <- nnf_linear(query, q_proj_weight, in_proj_bias[1:embed_dim])
      k <- nnf_linear(key, k_proj_weight, in_proj_bias[embed_dim:(embed_dim * 2)])
      v <- nnf_linear(value, v_proj_weight, in_proj_bias[(embed_dim*2):N])
    } else {
      q <- nnf_linear(query, q_proj_weight, in_proj_bias)
      k <- nnf_linear(key, k_proj_weight, in_proj_bias)
      v <- nnf_linear(value, v_proj_weight, in_proj_bias)
    }
    
  }
  
  q <- q * scaling
  
  if (!is.null(bias_k) && !is.null(bias_v)) {
    if (is.null(static_k) && is.null(static_v)) {
      k <- torch_cat(list(k, bias_k[["repeat"]](c(1, bsz, 1))))
      v <- torch_cat(list(v, bias_v[["repeat"]](c(1, bsz, 1))))
      
      if (!is.null(attn_mask))
        attn_mask <- nnf_pad(attn_mask, c(0,1))
      
      if (!is.null(key_padding_mask))
        key_padding_mask <- nnf_pad(key_padding_mask, c(0,1))
      
    }
  }
  
  q <- q$contiguous()$view(tgt_len, bsz * num_heads, head_dim)$transpose(0,1)
  
  if (!is.null(k))
    k <- k$contiguous()$view(-1, bsz * num_heads, head_dim)$transpose(0,1)
  
  if (!is.null(v))
    v <- v$contiguous()$view(-1, bsz * num_heads, head_dim)$transpose(0,1)
  
  
  if (!is.null(static_k))
    k <- static_k
  
  if (!is.null(static_v))
    v <- static_v
  
  src_len <- k$size(1)   
  
  if (add_zero_attn) {
    src_len <- src_len + 1
    k_size <- k$size()
    k <- torch_cat(list(k, torch_zeros(append(list(k_size[1], 1), k_size[3:length(k_size)]),
                                       dtype = k$dtype, device = k$device)), dim = 1)
    v_size
    k <- torch_cat(list(k, torch_zeros(append(list(v_size[1], 1), v_size[3:length(v_size)]),
                                       dtype = v$dtype, device = v$device)), dim = 1)
    
    if (!is.null(attn_mask)) {
      attn_mask <- nnf_pad(attn_mask, list(0,1))
    }
    
    if (!is.null(key_padding_mask)) {
      key_padding_mask <- nnf_pad(key_padding_mask, list(0,1))
    }
    
  }
  
  attn_output_weights <- torch_bmm(q, k$transpose(1, 2))
  
  if (!is.null(attn_mask)) {
    if (attn_mask$dtype == torch_bool()) {
      attn_output_weights$masked_fill_(attn_mask, -Inf)
    } else {
      attn_output_weights <- attn_output_weights + attn_mask
    }
  }
  
  if (!is.null(key_padding_mask)) {
    attn_output_weights <- attn_output_weights$view(vsz, num_heads, tgt_len, src_len)
    attn_output_weights <- attn_output_weights$masked_fill(
      key_padding_mask$unsqueeze(1)$unsqueeze(2),
      -Inf
    )
    attn_output_weights <- attn_output_weights$view(bsz * num_heads, tgt_len, 
                                                    src_len)
  }
  
  attn_output_weights <- nnf_softmax(attn_output_weights, dim=-1)
  attn_output_weights <- nnf_dropout(attn_output_weights, p=dropout_p, 
                                     training=training)
  
  attn_output <- torch_bmm(attn_output_weights, v)
  attn_output <- attn_output$transpose(0, 1)$contiguous()$view(tgt_len, bsz, embed_dim)
  attn_output <- nnf_linear(attn_output, out_proj_weight, out_proj_bias)
  
  if (need_weights) {
    attn_output_weights <- attn_output_weights$view(bsz, num_heads, tgt_len, 
                                                    src_len)
    return(list(attn_output, attn_output_weights$sum(dim = 1)/num_heads))
  } else {
    return(list(attn_output, NULL))
  }
}

nnf_multi_margin_loss <- function(input, target, p = 1, margin = 1, weight = NULL,
                                  reduction = "mean") {
  torch_multi_margin_loss(input, target, p, margin, weight, 
                          reduction = reduction_enum(reduction))
}

nnf_multilabel_margin_loss <- function(input, target, reduction = "mean") {
  torch_multilabel_margin_loss(input, target, reduction_enum(reduction))
}

nnf_multilabel_soft_margin_loss <- function(input, target, weight, reduction = "mean") {
  loss <- -(target * nnf_logsigmoid(input) + (1 - target) * nnf_logsigmoid(-input))
  
  if (!is.null(weight))
    loss <- loss * weight
  
  loss <- loss$sum(dim = 1)  / input$size(1)
  
  if (reduction == "none")
    ret <- loss
  else if (reduction == "mean")
    ret <- loss$mean()
  else if (reduction == "sum")
    ret <- loss$sum()
  else
    value_error("reduction is not valid.")
  
  ret
}

nnf_nll_loss <- function(input, target, weight = NULL, ignore_index, 
                         reduction = "mean") {
  
  dim <- input$dim()
  
  if (dim < 2)
    value_error("Expected 2 or more dimensions, got '{dim}'.")
  
  if (dim == 2)
    ret <- torch_nll_loss(input, target, weight, reduction_enum(reduction), 
                          ignore_index)
  else if (dim == 4)
    ret <- torch_nll_loss2d(input, target, weight, reduction_enum(reduction),
                            ignore_index)
  else {
    n <- input$size(0)
    c <- input$size(1)
    out_size <- c(n, input$size()[-c(1:2)])
    
    input <- input$contiguous()
    target <- target$contiguous()
    
    if (input$numel() > 0)
      input <- input$view(n, c, 1, -1)
    else
      input <- input$view(n, c, 0, 0)
    
    if (target$numel() > 0)
      target <- target$view(n, 1, -1)
    else
      target <- target$view(n, 0, 0)
    
    if (reduction != "none") {
      ret <- torch_nll_loss2d(input, target, weight, reduction_enum(reduction),
                              ignore_index)
    } else {
      out <- torch_nll_loss2d(input, target, weight, reduction_enum(reduction),
                              ignore_index)
      ret <- out$view(out_size)
    }
  }
  
  ret
}

nnf_normalize <- function(input, p = 2, dim = 1, eps = 1e-12, out = NULL) {
  if (is.null(out)) {
    denom <- input$norm(p, dim, keepdim = TRUE)$clamp_min(eps)$expand_as(input)
    return(input/denom)
  } else {
    denom <- input$norm(p, dim, keepdim=TRUE)$clamp_min_(eps)$expand_as(input)
    return(torch_div_out(input, denom, out))
  }
}

nnf_one_hot <- function(tensor, num_classes = -1) {
  torch_one_hot(tensor, num_classes)
}

nnf_pad_circular <- function(input, padding) {
  
  input <- torch_cat(list(input, input[,,1:tail(padding,1)]), dim = 2)
  input <- torch_cat(list(input[,,c(
    (-(rev(padding)[[1]] + rev(padding)[[2]])):(-rev(padding)[1])
  )],
  input), dim = 2)
  
  if (length(padding) > 2) {
    input <- torch_cat(list(input, input[,,,1:(rev(padding)[3])]), dim = 3)
    input <- torch_cat(list(input[,,,c(
      (-(rev(padding)[3] + rev(padding)[4])):(-rev(padding[3]))
    )],
    input), dim = 3)
  }
  
  if (length(padding) > 4) {
    input <- torch_cat(list(input, input[,,,,1:(rev(padding)[5])]), dim = 4)
    input <- torch_cat(list(input[,,,,c(
      (-(rev(padding)[5] + rev(padding)[6])):(-rev(padding[5]))
    )],
    input), dim = 4)
  }
  
  input
}

nnf_pad <- function(input, pad, mode = "constant", value = 0) {
  
  if (mode == "constant") {
    return(torch_constant_pad_nd(input, pad, value))
  } else {
    
    if (input$dim() == 3) {
      
      
      if (mode == "reflect")
        return(torch_reflection_pad1d(input, pad))
      
      if (mode == "replicate")
        return(torch_replication_pad1d(input, pad))
      
      if (mode == "circular")
        return(nnf_pad_circular(input, pad))
      
      not_implemented_error()
      
    } 
    
    if (input$dim() == 4) {
      
      if (mode == "reflect")
        return(torch_reflection_pad2d(input, pad))
      
      if (mode == "replicate")
        return(torch_replication_pad2d(input, pad))
      
      if (mode == "circular")
        return(nnf_pad_circular(input, pad))
      
      not_implemented_error()
    }
    
    if (input$dim() == 5) {
      
      if (mode == "reflect")
        not_implemented_error()
      
      if (mode == "replicate")
        return(torch_replication_pad3d(input, pad))
      
      if (mode == "circular")
        return(nnf_pad_circular(input, pad))
      
      
    }
    
  }
  
  not_implemented_error("Only 3D, 4D, 5D padding with non-constant padding are supported for now")
}

nnf_pairwise_distance <- function(x1, x2, p = 2, eps = 1e-6, keepdim = FALSE) {
  torch_pairwise_distance(x1, x2, p, eps, keepdim)
}

nnf_pdist <- function(input, p = 2) {
  torch_pdist(input, p)
}

nnf_pixel_shuffle <- function(input, upscale_factor) {
  torch_pixel_shuffle(input, upscale_factor)
}

nnf_poisson_nll_loss <- function(input, target, log_input = TRUE, full = FALSE, 
                                 eps = 1e-8, reduction = "mean") {
 torch_poisson_nll_loss(input, target, log_input, full, eps, 
                         reduction_enum(reduction))
}

nnf_prelu <- function(input, weight) {
  torch_prelu(input, weight)
}

nnf_relu <- function(input, inplace = FALSE) {
  if (inplace)
    torch_relu_(input)
  else
    torch_relu(input)
}

nnf_relu6 <- function(input, inplace = FALSE) {
  nnf_hardtanh(input, 0, 6, inplace)
}

nnf_relu_ <- function(input) {
  nnf_relu(input, inplace = TRUE)
}

nnf_rrelu <- function(input, lower = 1/8, upper = 1/3, training = FALSE,
                      inplace = FALSE) {
  
  if (inplace)
    result <- torch_rrelu_(input, lower, upper, training)
  else
    result <- torch_rrelu(input, lower, upper, training)
  
  result
}

nnf_rrelu_ <- function(input, lower = 1/8, upper = 1/3, training = FALSE) {
  nnf_rrelu(input, lower, upper, training = TRUE)
}

nnf_selu <- function(input, inplace = FALSE) {
  if (inplace)
    torch_selu_(input)
  else
    torch_selu(input)
}

nnf_selu_ <- function(input) {
  nnf_selu(input, inplace = TRUE)
}

nnf_smooth_l1_loss <- function(input, target, reduction = "mean") {
# def smooth_l1_loss(input, target, size_average=None, reduce=None, reduction='mean'):
#     # type: (Tensor, Tensor, Optional[bool], Optional[bool], str) -> Tensor
#     r"""Function that uses a squared term if the absolute
#     element-wise error falls below 1 and an L1 term otherwise.
# 
#     See :class:`~torch.nn.SmoothL1Loss` for details.
#     """
#     if not torch.jit.is_scripting():
#         tens_ops = (input, target)
#         if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
#             return handle_torch_function(
#                 smooth_l1_loss, tens_ops, input, target, size_average=size_average,
#                 reduce=reduce, reduction=reduction)
#     if not (target.size() == input.size()):
#         warnings.warn("Using a target size ({}) that is different to the input size ({}). "
#                       "This will likely lead to incorrect results due to broadcasting. "
#                       "Please ensure they have the same size.".format(target.size(), input.size()),
#                       stacklevel=2)
#     if size_average is not None or reduce is not None:
#         reduction = _Reduction.legacy_get_string(size_average, reduce)
  
  if (target$requires_grad) {
    ret <- nnf_smooth_l1_loss(input, target)
    if (reduction != "none") {
      if (reduction == "mean")
        ret <- torch_mean(ret)
      else
        ret <- torch_sum(ret)
    }
  } else {
    expanded <- torch_broadcast_tensors(list(input, target))
    ret <- torch_smooth_l1_loss(expanded[[1]], expanded[[2]], reduction_enum(reduction))
  }
  
  ret
}

nnf_soft_margin_loss <- function(input, target, reduction = "mean") {
  torch_soft_margin_loss(input, target, reduction_enum(reduction))
}

nnf_softmax <- function(input, dim, dtype = NULL) {
  if (is.null(dtype))
    ret <- input$softmax(dim)
  else
    ret <- input$softmax(dim, dtype = dtype)
  
  ret
}

nnf_softmin <- function(input, dim, dtype = NULL) {
  if (is.null(dtype))
    ret <- (-input)$softmax(dim)
  else
    ret <- (-input)$softmax(dim, dtype = dtype)
  
  ret
}

nnf_softplus <- function(input, beta = 1, threshold = 20) {
  torch_softplus(input, beta, threshold)
}

nnf_softshrink <- function(input, lambd = 0.5) {
  torch_softshrink(input, lambd)
}

nnf_softsign <- function(input) {
  input / (input$abs() + 1)
}

nnf_tanhshrink <- function(input) {
  input - input$tanh()
}

nnf_threshold <- function(input, threshold, value, inplace = FALSE) {
  if (inplace)
    torch_threshold_(input, threshold, value)
  else
    torch_threshold(input, threshold, value)
}

nnf_threshold_ <- function(input, threshold, value) {
  nnf_threshold(input, threshold, value, TRUE)
}

nnf_triplet_margin_loss <- function(anchor, positive, negative, margin = 1, p = 2,
                                    eps = 1e-6, swap = FALSE, reduction = "mean") {
  torch_triplet_margin_loss(anchor, positive, negative, margin, p, eps, swap,
                            reduction_enum(reduction))
}

nnf_unfold <- function(input, kernel_size, dilation = 1, padding = 0, stride = 1) {
  if (input$dim() == 4) {
    torch_im2col(input, nn_util_pair(kernel_size), nn_util_pair(dilation), 
                 nn_util_pair(padding), nn_util_pair(stride))
  } else {
    not_implemented_error("Input Error: Only 4D input Tensors are supported (got {input$dim()}D)")
  }
}
