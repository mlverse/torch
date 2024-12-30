#' Spam Data Loader
#'
#' A dataloader for the spam dataset used in machine learning. The dataset contains
#' 57 features extracted from email messages and a binary label indicating whether
#' an email is spam.
#'
#' @param url A character string representing the URL of the dataset.
#' @param batch_size Number of samples per batch.
#' @param shuffle Whether to shuffle the data.
#' @return A dataloader object.
#' @examples
#' dl <- spam_dataloader(batch_size = 32, shuffle = TRUE)
#' iter <- dl$.iter()
#' batch <- iter$.next()
#' @export
spam_dataloader <- function(url = "https://hastie.su.domains/ElemStatLearn/datasets/spam.data",
                            batch_size = 32, shuffle = TRUE) {
  # Load and preprocess data
  spam_data <- read.table(url, header = FALSE)
  x_data <- as.matrix(spam_data[, -ncol(spam_data)])
  y_data <- as.numeric(spam_data[, ncol(spam_data)]) - 1
  
  # Convert to tensors
  x_tensor <- torch_tensor(x_data, dtype = torch_float())
  y_tensor <- torch_tensor(y_data, dtype = torch_long())
  
  # Define dataset
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
  
  # Instantiate the dataset and dataloader
  dataset <- spam_dataset(x = x_tensor, y = y_tensor)
  dataloader(dataset, batch_size = batch_size, shuffle = shuffle)
}
