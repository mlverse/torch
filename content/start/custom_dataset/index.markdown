---
title: "Create your own Dataset"
weight: 3
description: | 
  Create a custom torch dataset.
---



Unless the data you're working with comes with some package in the `torch` ecosystem, you'll need to wrap in a `Dataset`.

# `torch` `Dataset` objects

A `Dataset` is an R6 object that knows how to iterate over data. This is because it acts as supplier to a `DataLoader` , who will ask it to return some number of items. How many? That is dependent on the batch size -- but batch sizes are handled by `DataLoaders`, so it needn't be concerned about that. All it has to know is what to do when asked for, e.g., item no. 7.

While a `Dataset` may have any number of methods -- each responsible for some aspect of pre-processing logic, for example -- just three methods are required:

-   `initialize()` , to pre-process and store the data;

-   `.getitem(i)`, to pick the item at position `i`, and

-   `.length()`, to indicate to the `DataLoader` how many items it has.

Let's see an example.

# Penguins

`penguins` is a very nice dataset that lives in the `palmerpenguins` CRAN package.


```r
library(dplyr)
library(palmerpenguins)

penguins %>% glimpse()
```

    Rows: 344
    Columns: 8
    $ species           <fct> Adelie, Adelie, Adelie, Adelie, Adelie, Adelie, Adelie, Adelie, Adelie, Adelie, Ade…
    $ island            <fct> Torgersen, Torgersen, Torgersen, Torgersen, Torgersen, Torgersen, Torgersen, Torger…
    $ bill_length_mm    <dbl> 39.1, 39.5, 40.3, NA, 36.7, 39.3, 38.9, 39.2, 34.1, 42.0, 37.8, 37.8, 41.1, 38.6, 3…
    $ bill_depth_mm     <dbl> 18.7, 17.4, 18.0, NA, 19.3, 20.6, 17.8, 19.6, 18.1, 20.2, 17.1, 17.3, 17.6, 21.2, 2…
    $ flipper_length_mm <int> 181, 186, 195, NA, 193, 190, 181, 195, 193, 190, 186, 180, 182, 191, 198, 185, 195,…
    $ body_mass_g       <int> 3750, 3800, 3250, NA, 3450, 3650, 3625, 4675, 3475, 4250, 3300, 3700, 3200, 3800, 4…
    $ sex               <fct> male, female, female, NA, female, male, female, male, NA, NA, NA, NA, female, male,…
    $ year              <int> 2007, 2007, 2007, 2007, 2007, 2007, 2007, 2007, 2007, 2007, 2007, 2007, 2007, 2007,…

There are three species, and we'll predict them from all available information: "biometrics" like `bill_length_mm`, geographic indicators like the `island` they live on, and more. The predictors are of two different types, categorical and continuous.

Continuous features, of R type `double`, may be fed to `torch` without further ado. We just directly use them to initialize a `torch` tensor, which will be of type `Float`:


```r
library(torch)
torch_tensor(1)
```

    torch_tensor
     1
    [ CPUFloatType{1} ]

It's different with categorical data though. Firstly, `torch` needs all data to be in numerical form, so vectors of type `character` need to become factors -- which can then be treated as numeric via level extraction. In the `penguins` dataset, `island`, `sex` , as well as the target column, `species`, are factors already. So can we just do an `as.numeric()` and that's it?

Not quite: We also need to reflect on the semantic side of things.

# Categorical data in deep learning

If we just replace islands *Biscoe*, *Dream*, and *Torgersen* by numbers 1, 2, and 3, we'd present them to the network as interval data, which of course they're not. We have two options: transform them to one-hot vectors, where e.g. *Biscoe* would be `0,0,1`, *Dream* `0,1,0`, and *Torgersen*, `1,0,0`, or leave them as they are, but have the network map each discrete value to a multidimensional, continuous representations. The latter is called embedding, and it often helps networks make sense of discrete data.

Embedding modules expect their inputs to be of type `Long`. A tensor created from an R value will have the correct type if make sure it's an `integer`:


```r
torch_tensor(as.integer(as.numeric(as.factor("one"))))
```


```r
torch_tensor
 1
[ CPULongType{1} ]
```

Now, let's create a dataset for penguins.

# A dataset for penguins

In `initialize()`, we convert the data as planned and store them for later delivery. Like the categorical input features, `species`, the target, is discrete, and thus, converted to `torch` `Long`.


```r
penguins_dataset <- dataset(
  
  name = "penguins_dataset",
  
  initialize = function(df) {
    
    df <- na.omit(df) 
    
    # continuous input data (x_cont)   
    x_cont <- df[ , c("bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "year")] %>%
      as.matrix()
    self$x_cont <- torch_tensor(x_cont)
    
    # categorical input data (x_cat)
    x_cat <- df[ , c("island", "sex")]
    x_cat$island <- as.integer(x_cat$island)
    x_cat$sex <- as.integer(x_cat$sex)
    self$x_cat <- as.matrix(x_cat) %>% torch_tensor()

    # target data (y)
    species <- as.integer(df$species)
    self$y <- torch_tensor(species)
    
  },
  
  .getitem = function(i) {
     list(x_cont = self$x_cont[i, ], x_cat = self$x_cat[i, ], y = self$y[i])
    
  },
  
  .length = function() {
    self$y$size()[[1]]
  }
 
)
```

Unlike `initialize`, `.getitem(i)` and `.length()` are just one-liners.

Let's see if this behaves like we want it to. We randomly split the data into training and validation sets and query their respective lengths:


```r
train_indices <- sample(1:nrow(penguins), 250)

train_ds <- penguins_dataset(penguins[train_indices, ])
valid_ds <- penguins_dataset(penguins[setdiff(1:nrow(penguins), train_indices), ])

length(train_ds)
length(valid_ds)
```

    [1] 242
    [1] 91

We can index into `Dataset`s in an R-like way:


```r
train_ds[1]
```

    $x_cont
    torch_tensor
       45.2000
       16.4000
      223.0000
     5950.0000
     2008.0000
    [ CPUFloatType{5} ]

    $x_cat
    torch_tensor
     1
     2
    [ CPULongType{2} ]

    $y
    torch_tensor
    3
    [ CPULongType{} ]

From here on, everything proceeds like in the first tutorial: We use the `Dataset`s to instantiate `DataLoader`s...


```r
train_dl <- train_ds %>% dataloader(batch_size = 16, shuffle = TRUE)

valid_dl <- valid_ds %>% dataloader(batch_size = 16, shuffle = FALSE)
```

... and then, create and train the network. The network will look pretty different now though: most notably, you'll see embeddings at work.

# Classifying penguins -- the network

We just heard that embedding layers work with a datatype that's different from most other neural network layers. It is therefore convenient to have them work in a space of their own, that is, put them into a dedicated container.

Here we define a specialized module that has one embedding layer for every categorical feature. It gets passed the cardinalities of the respective features, and creates an `nn_embedding()` for each of them.

When called, it iterates over its submodules, lets them do their work, and returns the concatenated output.


```r
embedding_module <- nn_module(
  
  initialize = function(cardinalities) {
    
    self$embeddings = nn_module_list(lapply(cardinalities, function(x) nn_embedding(num_embeddings = x, embedding_dim = ceiling(x/2))))
    
  },
  
  forward = function(x) {
    
    embedded <- vector(mode = "list", length = length(self$embeddings))
    for (i in 1:length(self$embeddings)) {
      embedded[[i]] <- self$embeddings[[i]](x[ , i])
    }
    
    torch_cat(embedded, dim = 2)
  }
)
```

The top-level module has three submodules: said `embedding_module` and two linear layers.

The first linear layer takes the output from `embedding_module` , computes the affine transformation it sees fit, and passes its result to the output layer. `output` then has three units, one for every possible target class.

The activation function we apply to the raw aggregation, `nnf_log_softmax()`, composes two operations: the popular-in-deep-learning `softmax` normalization algorithm and taking the log. Like that, we end up with the format expected by `nnf_nll_loss()`, the loss function that computes the negative log likelihood (NLL) loss between inputs and targets.


```r
net <- nn_module(
  "penguin_net",

  initialize = function(cardinalities,
                        n_cont,
                        fc_dim,
                        output_dim) {
    
    self$embedder <- embedding_module(cardinalities)
    self$fc1 <- nn_linear(sum(purrr::map(cardinalities, function(x) ceiling(x/2)) %>% unlist()) + n_cont, fc_dim)
    self$output <- nn_linear(fc_dim, output_dim)
    
  },

  forward = function(x_cont, x_cat) {
    
    embedded <- self$embedder(x_cat)
    
    all <- torch_cat(list(embedded, x_cont$to(dtype = torch_float())), dim = 2)
    
    all %>% self$fc1() %>%
      nnf_relu() %>%
      self$output() %>%
      nnf_log_softmax(dim = 2)
    
  }
)
```

Let's instantiate the top-level module:


```r
model <- net(
  cardinalities = c(length(levels(penguins$island)), length(levels(penguins$sex))),
  n_cont = 5,
  fc_dim = 32,
  output_dim = 3
)
```

And we're ready for training!

# Model training


```r
optimizer <- optim_adam(model$parameters, lr = 0.01)

for (epoch in 1:20) {

  model$train()
  train_losses <- c()  

  for (b in enumerate(train_dl)) {
    
    optimizer$zero_grad()
    output <- model(b$x_cont, b$x_cat)
    loss <- nnf_nll_loss(output, b$y)
    
    loss$backward()
    optimizer$step()
    
    train_losses <- c(train_losses, loss$item())
    
  }

  model$eval()
  valid_losses <- c()

  for (b in enumerate(valid_dl)) {
    
    output <- model(b$x_cont, b$x_cat)
    loss <- nnf_nll_loss(output, b$y)
    valid_losses <- c(valid_losses, loss$item())
    
  }

  cat(sprintf("Loss at epoch %d: training: %3.3f, validation: %3.3f\n", epoch, mean(train_losses), mean(valid_losses)))
}
```


    Loss at epoch 1: training: 34.962, validation: 4.354
    Loss at epoch 2: training: 8.207, validation: 14.512
    Loss at epoch 3: training: 7.804, validation: 2.820
    Loss at epoch 4: training: 5.998, validation: 8.525
    Loss at epoch 5: training: 8.293, validation: 5.594
    Loss at epoch 6: training: 6.375, validation: 4.540
    Loss at epoch 7: training: 7.478, validation: 2.120
    Loss at epoch 8: training: 3.470, validation: 3.508
    Loss at epoch 9: training: 12.155, validation: 4.266
    Loss at epoch 10: training: 10.168, validation: 4.285
    Loss at epoch 11: training: 5.963, validation: 1.888
    Loss at epoch 12: training: 3.035, validation: 2.454
    Loss at epoch 13: training: 1.993, validation: 1.185
    Loss at epoch 14: training: 2.454, validation: 2.200
    Loss at epoch 15: training: 1.641, validation: 0.588
    Loss at epoch 16: training: 0.996, validation: 1.959
    Loss at epoch 17: training: 0.912, validation: 0.674
    Loss at epoch 18: training: 1.517, validation: 0.487
    Loss at epoch 19: training: 1.569, validation: 1.202
    Loss at epoch 20: training: 0.735, validation: 1.313`
