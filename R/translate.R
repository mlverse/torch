translate_error_msg <- function(msg) {
  msg %>% 
    translate_dim_size_error_msg() %>% 
    translate_dim_error_msg() %>% 
    translate_max_index_msg() %>% 
    translate_index_out_of_range_msg() %>% 
    translate_target_index_msg() %>% 
    translate_contract_error_msg() %>% 
    translate_null_index_error_msg() %>%
    translate_tensor_number_error_msg() %>%
    translate_size_match_error_msg()
}

translate_dim_error_msg <- function(msg) {
  regex <- "(?:.|\\r?\\n)*Dimension out of range \\(expected to be in range of \\[-[0-9]+, ([0-9]+)\\], but got (-?[0-9]+)\\)(?:.|\\r?\\n)*"
  translate_increase_group(msg, regex)
}

translate_dim_size_error_msg <- function(msg) {
  regex <- "(?:.|\\r?\\n)*dimension ([0-9]+) does not have size [0-9]+(?:.|\\r?\\n)*"
  translate_increase_group(msg, regex)
}

translate_tensor_number_error_msg <- function(msg) {
  regex <- "(?:.|\\r?\\n)*Expected size [0-9]+ but got size [0-9]+ for tensor number ([0-9]+) in the list(?:.|\\r?\\n)*"
  translate_increase_group(msg, regex)
}

translate_size_match_error_msg <- function(msg) {
  regex <- "(?:.|\\r?\\n)*Sizes of tensors must match except in dimension ([0-9]+)(?:.|\\r?\\n)*"
  translate_increase_group(msg, regex)
}

translate_max_index_msg <- function(msg) {
  regex <- "(?:.|\\r?\\n)*Found an invalid max index: ([0-9]+)(?:.|\\r?\\n)*"
  translate_increase_group(msg, regex)
}

translate_index_out_of_range_msg <- function(msg) {
  regex <- "(?:.|\\r?\\n)*index ([0-9]+) out of range for tensor of size \\[[0-9]+\\] at dimension ([0-9]+)(?:.|\\r?\\n)*"
  translate_increase_group(msg, regex)
}

translate_target_index_msg <- function(msg) {
  regex <- "(?:.|\\r?\\n)*Target (-?[0-9]+) is out of bounds.(?:.|\\r?\\n)*"
  translate_increase_group(msg, regex, ifpos = FALSE)
}

translate_contract_error_msg <- function(msg) {
  regex <- "(?:.|\\r?\\n)*contracted dimensions need to match, but first has size [0-9]+ in dim ([0-9]+) and second has size [0-9]+ in dim ([0-9]+)(?:.|\\r?\\n)*"
  translate_increase_group(msg, regex)
}

translate_null_index_error_msg <- function(msg) {
  regex <- "(?:.|\\r?\\n)*index (-?[0-9]+) is out of bounds for dimension ([0-9]+) with size 0"
  translate_increase_group(msg, regex)
}

regexec2 <- function(regex, msg) {
  out <- regexec(regex, msg)
  atr <- attributes(out[[1]])
  out[[1]] <- out[[1]][-1]
  atr$match.length <- atr$match.length[-1]
  attributes(out[[1]]) <- atr
  out
}

translate_increase_group <- function(msg, regex, ifpos=TRUE) {
  if (!grepl(regex, msg))
    return(msg)
  
  values <- regmatches(msg, regexec2(regex, msg))
  values <- lapply(values, function(x) {
    x <- as.integer(x)
    if (ifpos) 
      as.character(ifelse(x >= 0, x + 1, x))
    else
      as.character(x+1)
  })
  
  regmatches(msg, regexec2(regex, msg)) <- values
  msg
}