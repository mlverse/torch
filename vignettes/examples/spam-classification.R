library(torch)

# Create the spam dataset
spam_data <- spam_dataset()
cat("Dataset size:", spam_data$.length(), "\n")

# Create train/test split
test_size <- 0.2
n <- spam_data$.length()
indices <- torch_randperm(n) + 1
split_idx <- floor(n * (1 - test_size))

# Create training and test datasets
train_indices <- indices[1:split_idx]
test_indices <- indices[(split_idx+1):n]

# Create data loaders
train_dl <- dataloader(spam_data, batch_size = 64, shuffle = FALSE, 
                       sampler = torch::sampler(train_indices))
test_dl <- dataloader(spam_data, batch_size = 64, shuffle = FALSE, 
                      sampler = torch::sampler(test_indices))

# Create a simple neural network for spam classification
spam_classifier <- nn_module(
  "SpamClassifier",
  initialize = function() {
    self$fc1 <- nn_linear(57, 32)
    self$fc2 <- nn_linear(32, 16)
    self$fc3 <- nn_linear(16, 1)
    self$dropout <- nn_dropout(0.2)
  },
  forward = function(x) {
    x %>% 
      self$fc1() %>% 
      nnf_relu() %>% 
      self$dropout() %>%
      self$fc2() %>% 
      nnf_relu() %>% 
      self$dropout() %>%
      self$fc3() %>% 
      torch_sigmoid()
  }
)

# Initialize model and optimizer
model <- spam_classifier()
optimizer <- optim_adam(model$parameters, lr = 0.001)
loss_fn <- nn_bce_loss()

# Training function
train_epoch <- function(model, dataloader, optimizer, loss_fn) {
  model$train()
  total_loss <- 0
  n_batches <- 0
  
  coro::loop(for (batch in dataloader) {
    optimizer$zero_grad()
    
    # Get inputs and labels
    inputs <- batch[[1]]
    labels <- batch[[2]]$unsqueeze(2)$float()
    
    # Forward pass
    outputs <- model(inputs)
    loss <- loss_fn(outputs, labels)
    
    # Backward pass and optimize
    loss$backward()
    optimizer$step()
    
    total_loss <- total_loss + loss$item()
    n_batches <- n_batches + 1
  })
  
  return(total_loss / n_batches)
}

# Evaluation function
evaluate <- function(model, dataloader) {
  model$eval()
  correct <- 0
  total <- 0
  
  with_no_grad({
    coro::loop(for (batch in dataloader) {
      inputs <- batch[[1]]
      labels <- batch[[2]]
      
      outputs <- model(inputs)
      predictions <- (outputs > 0.5)$squeeze()
      
      total <- total + labels$size(1)
      correct <- correct + (predictions == labels)$sum()$item()
    })
  })
  
  return(correct / total)
}

# Train for several epochs
num_epochs <- 10

for (epoch in 1:num_epochs) {
  # Train
  avg_loss <- train_epoch(model, train_dl, optimizer, loss_fn)
  
  # Evaluate
  if (epoch %% 2 == 0) {
    train_acc <- evaluate(model, train_dl)
    test_acc <- evaluate(model, test_dl)
    cat(sprintf("Epoch %d/%d, Loss: %.4f, Train Acc: %.2f%%, Test Acc: %.2f%%\n", 
                epoch, num_epochs, avg_loss, train_acc * 100, test_acc * 100))
  } else {
    cat(sprintf("Epoch %d/%d, Loss: %.4f\n", epoch, num_epochs, avg_loss))
  }
}

# Final evaluation
train_acc <- evaluate(model, train_dl)
test_acc <- evaluate(model, test_dl)
cat(sprintf("Final Results - Train Acc: %.2f%%, Test Acc: %.2f%%\n", 
            train_acc * 100, test_acc * 100))