# dataset

``` r
library(torch)

# In deep learning models you don't usually have all your data in RAM
# because you are usually training using mini-batch gradient descent
# thus only needing a mini-batch on RAM each time.

# In torch we use the `datasets` abstraction to define the process of
# loading data. Once you have defined your dataset you can use torch
# dataloaders that allows you to iterate over this dataset in batches.

# Note that datasets are optional in torch. They are jut there as a
# recommended way to load data.

# Below you will see an example of how to create a simple torch dataset
# that pre-process a data.frame into tensors so you can feed them to
# a model.

df_dataset <- dataset(
  "mydataset",

  # the input data to your dataset goes in the initialize function.
  # our dataset will take a dataframe and the name of the response
  # variable.
  initialize = function(df, response_variable) {
    self$df <- df[,-which(names(df) == response_variable)]
    self$response_variable <- df[[response_variable]]
  },

  # the .getitem method takes an index as input and returns the
  # corresponding item from the dataset.
  # the index could be anything. the dataframe could have many
  # rows for each index and the .getitem method would do some
  # kind of aggregation before returning the element.
  # in our case the index will be a row of the data.frame,
  .getitem = function(index) {
    response <- torch_tensor(self$response_variable[index])
    x <- torch_tensor(as.numeric(self$df[index,]))

    # note that the dataloaders will automatically stack tensors
    # creating a new dimension
    list(x = x, y = response)
  },

  # It's optional, but helpful to define the .length method returning
  # the number of elements in the dataset. This is needed if you want
  # to shuffle your dataset.
  .length = function() {
    length(self$response_variable)
  }

)


# we can now initialize an instance of our dataset.
# for example
mtcars_dataset <- df_dataset(mtcars, "mpg")

# now we can get an item with
mtcars_dataset$.getitem(1)
```

    ## $x
    ## torch_tensor
    ##    6.0000
    ##  160.0000
    ##  110.0000
    ##    3.9000
    ##    2.6200
    ##   16.4600
    ##    0.0000
    ##    1.0000
    ##    4.0000
    ##    4.0000
    ## [ CPUFloatType{10} ]
    ## 
    ## $y
    ## torch_tensor
    ##  21
    ## [ CPUFloatType{1} ]

``` r
# Given a dataset you can create a dataloader with
dl <- dataloader(mtcars_dataset, batch_size = 15, shuffle = TRUE)

# we can then loop trough the elements of the dataloader with
coro::loop(for(batch in dl) {
  cat("X size:  ")
  print(batch[[1]]$size())
  cat("Y size:  ")
  print(batch[[2]]$size())
})
```

    ## X size:  [1] 15 10
    ## Y size:  [1] 15  1
    ## X size:  [1] 15 10
    ## Y size:  [1] 15  1
    ## X size:  [1]  2 10
    ## Y size:  [1] 2 1
