# Loading data

``` r
library(torch)
```

## Datasets and data loaders

Central to data ingestion and preprocessing are datasets and data
loaders. A dataset is an object that holds the data to use, while a data
loader is an object that will load the data from a dataset providing a
way to access subsets of the data. By using datasets and data loaders
you will have a process for clearly organizing your data and passing it
to other components of the torch package, such as model training.

Built into `torch` are premade datasets that are commonly used in
machine learning, such as the MNIST handwriting dataset
(`mnist_dataset()`). Most of the prebuilt datasets relate to image
recognition and natural language processing.

Below is an example of how you would use the MNIST dataset with a
dataloader. First, the `minst_dataset()` function is used to create `ds`
which is a `Dataset` object. Then a dataloader `dl` is created to query
that data. Finally, that dataloader is used in a
[`coro::loop()`](https://coro.r-lib.org/reference/collect.html) to
iterate over batches of that data:

``` r
# Create a dataset from included data
ds <- mnist_dataset(
  dir, 
  download = TRUE, 
  transform = function(x) {
    x <- x$to(dtype = torch_float())/256
    x[newaxis,..]
  }
)

# Create the loader to query the data in batches
dl <- dataloader(ds, batch_size = 32, shuffle = TRUE)

coro::loop(for (b in dl)) {
# use the data from each batch `b` here
# ...
})
```

See `vignettes/examples/mnist-cnn.R` for a complete example.

In the more common situation where you have a unique set of data that
isn’t included with the package you’ll need to make a custom `Dataset`
subclass by using the
[`dataset()`](https://torch.mlverse.org/docs/dev/reference/dataset.md)
function. The custom `Dataset` subclass is an abstract R6 container for
the data. It will need to know some information about the particular
dataset, such as how to iterate over it.

At a minimum, when using
[`dataset()`](https://torch.mlverse.org/docs/dev/reference/dataset.md)
to create a custom `Dataset` class you’ll want to define the following:

- `name` - for convenience, keep track of what type of data it is
- `initialize` - a member function defining how to create a object with
  that class. It could have no parameters, for when all objects of that
  class will be the same, or you can give it specific parameters usually
  for if different objects should have different data.
- `.getitem` - this member function is called when the dataloader goes
  to pull a new batch of data. You can include preprocessing in this
  function if needed. Note that the function will be called extremely
  frequently, so it’s advantageous to make it fast.
- `.length` - this will return the amount of data in the dataset, which
  is helpful for users.

## Example of using a custom Dataset

While this may sound complicated the base logic is only a few steps–the
complexity often comes from the data itself and how involved your
preprocessing is. Here we show how to create your own `Dataset` class to
train on [Allison Horst's
penguins](https://github.com/allisonhorst/palmerpenguins).

| Component       |                                                                                                                                `Dataset` R6 class                                                                                                                                 |                                 `Dataset` object                                 |                       `DataLoader` object                        |                           batch                           |
|:----------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------:|:----------------------------------------------------------------:|:---------------------------------------------------------:|
| Description     | Output of [`dataset()`](https://torch.mlverse.org/docs/dev/reference/dataset.md). When calling [`dataset()`](https://torch.mlverse.org/docs/dev/reference/dataset.md) it should have at least a `name`, `initialize`, `.getitem`, and `.length`. Output is a `Dataset` generator. | Object created by using the custom `Dataset` generator. Actually stores the data | Object that queries the `Dataset` object to pull batches of data | The subsample of data used for things like model training |
| Penguin example |                                                                                                                                `penguins_dataset`                                                                                                                                 |                                     `tuxes`                                      |                               `dl`                               |                            `b`                            |

``` r
library(palmerpenguins)
#> 
#> Attaching package: 'palmerpenguins'
#> The following objects are masked from 'package:datasets':
#> 
#>     penguins, penguins_raw
library(magrittr)

penguins
#> # A tibble: 344 × 8
#>    species island    bill_length_mm bill_depth_mm flipper_length_mm body_mass_g
#>    <fct>   <fct>              <dbl>         <dbl>             <int>       <int>
#>  1 Adelie  Torgersen           39.1          18.7               181        3750
#>  2 Adelie  Torgersen           39.5          17.4               186        3800
#>  3 Adelie  Torgersen           40.3          18                 195        3250
#>  4 Adelie  Torgersen           NA            NA                  NA          NA
#>  5 Adelie  Torgersen           36.7          19.3               193        3450
#>  6 Adelie  Torgersen           39.3          20.6               190        3650
#>  7 Adelie  Torgersen           38.9          17.8               181        3625
#>  8 Adelie  Torgersen           39.2          19.6               195        4675
#>  9 Adelie  Torgersen           34.1          18.1               193        3475
#> 10 Adelie  Torgersen           42            20.2               190        4250
#> # ℹ 334 more rows
#> # ℹ 2 more variables: sex <fct>, year <int>
```

In addition, any number of helper functions can be defined.

Here, we assume the `penguins` have already been loaded, and all
preprocessing consists in removing lines with `NA` values, transforming
`factor`s to numbers starting from 0, and converting from R data types
to `torch` tensors.

In `.getitem`, we essentially decide how this data is going to be used:
All variables besides `species` go into `x`, the predictor, and
`species` will constitute `y`, the target. Predictor and target are
returned in a list, to be accessed as `batch[[1]]` and `batch[[2]]`
during training.

``` r
penguins_dataset <- dataset(
  
  name = "penguins_dataset",
  
  initialize = function() {
    self$data <- self$prepare_penguin_data()
  },
  
  .getitem = function(index) {
    
    x <- self$data[index, 2:-1]
    y <- self$data[index, 1]$to(torch_long())
    
    list(x, y)
  },
  
  .length = function() {
    self$data$size()[[1]]
  },
  
  prepare_penguin_data = function() {
    
    input <- na.omit(penguins) 
    # conveniently, the categorical data are already factors
    input$species <- as.numeric(input$species)
    input$island <- as.numeric(input$island)
    input$sex <- as.numeric(input$sex)
    
    input <- as.matrix(input)
    torch_tensor(input)
  }
)
```

Let’s create the dataset , query for it’s length, and look at its first
item:

``` r
tuxes <- penguins_dataset()
tuxes$.length()
#> [1] 333
tuxes$.getitem(1)
#> [[1]]
#> torch_tensor
#>     3.0000
#>    39.1000
#>    18.7000
#>   181.0000
#>  3750.0000
#>     2.0000
#>  2007.0000
#> [ CPUFloatType{7} ]
#> 
#> [[2]]
#> torch_tensor
#> 1
#> [ CPULongType{} ]
```

To be able to iterate over `tuxes`, we need a data loader (we override
the default batch size of `1`):

``` r
dl <- tuxes %>% dataloader(batch_size = 8)
```

Calling `.length()` on a data loader (as opposed to a dataset) will
return the number of `batches` we have:

``` r
dl$.length()
#> [1] 42
```

And we can create an iterator to inspect the first batch:

``` r
iter <- dl$.iter()
b <- iter$.next()
b
#> [[1]]
#> torch_tensor
#>     3.0000    39.1000    18.7000   181.0000  3750.0000     2.0000  2007.0000
#>     3.0000    39.5000    17.4000   186.0000  3800.0000     1.0000  2007.0000
#>     3.0000    40.3000    18.0000   195.0000  3250.0000     1.0000  2007.0000
#>     3.0000    36.7000    19.3000   193.0000  3450.0000     1.0000  2007.0000
#>     3.0000    39.3000    20.6000   190.0000  3650.0000     2.0000  2007.0000
#>     3.0000    38.9000    17.8000   181.0000  3625.0000     1.0000  2007.0000
#>     3.0000    39.2000    19.6000   195.0000  4675.0000     2.0000  2007.0000
#>     3.0000    41.1000    17.6000   182.0000  3200.0000     1.0000  2007.0000
#> [ CPUFloatType{8,7} ]
#> 
#> [[2]]
#> torch_tensor
#>  1
#>  1
#>  1
#>  1
#>  1
#>  1
#>  1
#>  1
#> [ CPULongType{8} ]
```

To train a network, we can use
[`coro::loop()`](https://coro.r-lib.org/reference/collect.html) to
iterate over batches.

### Training with data loaders

Our example network is very simple. (In reality, we would want to treat
`island` as the categorical variable it is, and either one-hot-encode or
embed it.)

``` r
net <- nn_module(
  "PenguinNet",
  initialize = function() {
    self$fc1 <- nn_linear(7, 32)
    self$fc2 <- nn_linear(32, 3)
  },
  forward = function(x) {
    x %>% 
      self$fc1() %>% 
      nnf_relu() %>% 
      self$fc2() %>% 
      nnf_log_softmax(dim = 1)
  }
)

model <- net()
```

We still need an optimizer:

``` r
optimizer <- optim_sgd(model$parameters, lr = 0.01)
```

And we’re ready to train:

``` r
for (epoch in 1:10) {
  
  l <- c()
  
  coro::loop(for (b in dl) {
    optimizer$zero_grad()
    output <- model(b[[1]])
    loss <- nnf_nll_loss(output, b[[2]])
    loss$backward()
    optimizer$step()
    l <- c(l, loss$item())
  })
  
  cat(sprintf("Loss at epoch %d: %3f\n", epoch, mean(l)))
}
#> Loss at epoch 1: 40.393332
#> Loss at epoch 2: 2.068251
#> Loss at epoch 3: 2.068251
#> Loss at epoch 4: 2.068251
#> Loss at epoch 5: 2.068251
#> Loss at epoch 6: 2.068251
#> Loss at epoch 7: 2.068251
#> Loss at epoch 8: 2.068251
#> Loss at epoch 9: 2.068251
#> Loss at epoch 10: 2.068251
```

Through this example we have trained a deep learning model using
[`dataset()`](https://torch.mlverse.org/docs/dev/reference/dataset.md)
to define a custom class and then loaded it in batches with a data
loader. By using the dataset and data loader we were able to write code
that split the data preprocessing and setup from the model training
itself.

## Notes on efficiency

When using datasets and data loaders you may find that under certain
conditions your code is running more slowly than you’d expect. In some
situations the overhead of using dataloaders and datasets can impact
overall performance. This may change in time as the R/C++ integration of
Torch improves, but for now there are some workarounds:

### Use `.getbatch()` instead of `.getitem()`

By default a dataloader will use the `.getitem()` member function to
pull each single datapoint individually. You can speed this up by
switching to using `.getbatch()` which will pull all the datapoints in a
batch at once:

``` r
penguins_dataset_batching <- dataset(
  
  name = "penguins_dataset_batching",
  
  initialize = function() {
    self$data <- self$prepare_penguin_data()
  },
  
  # the only change is that this went from .getitem to .getbatch
  .getbatch = function(index) {
    
    x <- self$data[index, 2:-1]
    y <- self$data[index, 1]$to(torch_long())
    
    list(x, y)
  },
  
  .length = function() {
    self$data$size()[[1]]
  },
  
  prepare_penguin_data = function() {
    
    input <- na.omit(penguins) 
    # conveniently, the categorical data are already factors
    input$species <- as.numeric(input$species)
    input$island <- as.numeric(input$island)
    input$sex <- as.numeric(input$sex)
    
    input <- as.matrix(input)
    torch_tensor(input)
  }
)
```

In many instances the only change is to exactly replace just `.getitem`
with `.getbatch` since often the `.getitem` function is written to
handle vectors of indices. In this penguins example the `.getitem`
function used the index to select the rows, which will work fine with a
vector instead

### Remove dataset dataloader and manually define the function calls

If switching to `.getbatch` does not provide the benefit you were
expecting you could also remove the `dataset` entirely and manually pass
the data. At this point you are trading readability of your code and
convenience for speed.

``` r
input <- na.omit(penguins) 
# conveniently, the categorical data are already factors
input$species <- as.numeric(input$species)
input$island <- as.numeric(input$island)
input$sex <- as.numeric(input$sex)

input <- as.matrix(input)
input <- torch_tensor(input)

data_x <- input[, 2:-1]
data_y <- input[, 1]$to(torch_long())

batch_size <- 8
num_data_points <- data_y$size(1)
num_batches <- floor(num_data_points/batch_size)

for(epoch in 1:10){

  # rearrange the data each epoch
  permute <- torch_randperm(num_data_points) + 1L
  data_x <- data_x[permute]
  data_y <- data_y[permute]
  
  # manually loop through the batches
  for(batch_idx in 1:num_batches){

    # here index is a vector of the indices in the batch
    index <- (batch_size*(batch_idx-1) + 1):(batch_idx*batch_size)
    
    x <- data_x[index]
    y <- data_y[index]$to(torch_long())

    optimizer$zero_grad()
    output <- model(x)
    loss <- nnf_nll_loss(output, y)
    loss$backward()
    optimizer$step()
    l <- c(l, loss$item())
  }
  
  cat(sprintf("Loss at epoch %d: %3f\n", epoch, mean(l)))
}
```
