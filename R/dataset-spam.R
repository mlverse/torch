#' Spam Email Classification Dataset
#'
#' Dataset from "Elements of Statistical Learning" containing email characteristics 
#' with 57 features and a binary classification (spam=1, not-spam=0).
#' 
#' The dataset contains the following features:
#' - 57 continuous predictors representing various characteristics of emails
#' - Binary outcome: 0 (not spam) or 1 (spam)
#'
#' @param url URL of the dataset, defaults to the standard location from Elements of Statistical Learning website
#' @param download Whether to download the dataset. If FALSE, you must provide the data yourself
#' @param transform An optional function that takes a tensor and returns a transformed version
#'
#' @return A dataset object that can be used with a dataloader
#' @export
#'
#' @examples
#' \dontrun{
#' # Create the dataset
#' ds <- spam_dataset()
#' 
#' # Get dataset size
#' ds$.length()
#' 
#' # Get a single sample
#' sample <- ds$.getitem(1)
#' 
#' # Create a dataloader
#' dl <- dataloader(ds, batch_size = 32, shuffle = TRUE)
#' 
#' # Iterate through batches
#' coro::loop(for (batch in dl) {
#'   x <- batch[[1]]  # Features (57 dimensions)
#'   y <- batch[[2]]  # Target (0 = not spam, 1 = spam)
#'   # ... process the batch
#' })
#' }
spam_dataset <- function(url = "https://hastie.su.domains/ElemStatLearn/datasets/spam.data", 
                         download = TRUE,
                         transform = NULL) {
  
  # Create a closure that holds the data and provides the required methods
  env <- new.env(parent = emptyenv())
  
  # Download and prepare data
  if (download) {
    # Download the data
    temp_file <- tempfile()
    tryCatch({
      download.file(url, temp_file, mode = "wb")
    }, error = function(e) {
      stop(paste("Failed to download the spam dataset:", e$message))
    })
    
    # Read the data - space-separated values, no header
    tryCatch({
      spam_data <- read.table(temp_file, header = FALSE)
    }, error = function(e) {
      stop(paste("Failed to parse the spam dataset:", e$message))
    })
    
    # Convert to tensor
    input <- as.matrix(spam_data)
    env$data <- torch::torch_tensor(input, dtype = torch::torch_float32())
  } else {
    stop("Non-download mode not implemented. Please set download=TRUE")
  }
  
  env$transform <- transform
  
  # Create a list with the required methods
  dataset <- list(
    .getitem = function(index) {
      # Get features (all columns except the last one)
      x <- env$data[index, 1:57]
      
      # Apply transformation if provided
      if (!is.null(env$transform))
        x <- env$transform(x)
      
      # Get target (last column), convert to long tensor for classification
      # Extract the item value and create a new scalar tensor to ensure proper shape
      label_value <- env$data[index, 58]$item()
      y <- torch::torch_tensor(label_value, dtype = torch::torch_long())
      
      list(x, y)
    },
    
    .length = function() {
      env$data$size()[[1]]
    }
  )
  
  # Make sure the class attribute is set correctly for torch to recognize
  class(dataset) <- c("spam_dataset", "dataset", "R6")
  
  dataset
}