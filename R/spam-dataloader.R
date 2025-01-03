#' Spam Data Loader
#'
#' A dataloader for the spam dataset commonly used in machine learning. The dataset
#' contains 57 features extracted from email messages and a binary label indicating
#' whether an email is spam (1) or not spam (0).
#'
#' @param url A character string representing the URL of the dataset. Defaults to
#' "https://hastie.su.domains/ElemStatLearn/datasets/spam.data".
#' @param batch_size Number of samples per batch. Defaults to 32.
#' @param shuffle Logical; whether to shuffle the data. Defaults to TRUE.
#' @param download Logical; whether to download the dataset if not already available. Defaults to FALSE.
#' @return A dataloader object for the spam dataset.
#' @examples
#' dl <- spam_dataloader(batch_size = 32, shuffle = TRUE)
#' iter <- dl$.iter()
#' batch <- iter$.next()
#' print(batch)
#' @export
spam_dataloader <- function(url = "https://hastie.su.domains/ElemStatLearn/datasets/spam.data",
                            batch_size = 32, shuffle = TRUE, download = FALSE) {
  library(torch)  # Ensure torch is loaded
  
  # Download the dataset if needed
  data_path <- tempfile(fileext = ".data")
  if (download) {
    download.file(url, data_path)
  } else {
    data_path <- url
  }
  
  # Load and preprocess the dataset
  spam_data <- read.table(data_path, header = FALSE)
  x_data <- as.matrix(spam_data[, -ncol(spam_data)])  # Extract predictors
  y_data <- as.numeric(spam_data[, ncol(spam_data)]) - 1  # Extract target (convert to 0/1)
  
  # Convert data to tensors
  x_tensor <- torch_tensor(x_data, dtype = torch_float())
  y_tensor <- torch_tensor(y_data, dtype = torch_long())
  
  # Define the dataset class
  spam_dataset <- dataset(
    name = "spam_dataset",
    initialize = function(x, y) {
      self$x <- x
      self$y <- y
    },
    .getbatch = function(index) {
      list(
        x = self$x[index, ],
        y = self$y[index]
      )
    },
    .length = function() {
      self$y$size(1)
    }
  )
  
  # Create the dataset and dataloader
  dataset <- spam_dataset(x = x_tensor, y = y_tensor)
  dataloader(dataset, batch_size = batch_size, shuffle = shuffle)
}
