---
title: "Create your own Dataset"
weight: 3
description: | 
  Create a custom torch dataset.
---

Unless the data you're working with comes with some package in the `torch` ecosystem, you'll need to wrap in a `Dataset`.

# `torch` `Dataset` objects

A `Dataset` is an R6 object that knows how to iterate over data. This is because acts as supplier to a `DataLoader` , who will ask it to return some number of items. How many? That is dependent on the batch size -- but batch sizes are handled by `DataLoaders`, so it needn't be concerned about that. All it has to know is what to do when asked for, e.g., items 9-16.

While a `Dataset` may have any number of methods -- each responsible for some aspect of pre-processing logic, for example --


```r
library(torch)
library(palmerpenguins)

penguins
```

```
## # A tibble: 344 x 8
##    species island bill_length_mm bill_depth_mm flipper_length_… body_mass_g
##    <fct>   <fct>           <dbl>         <dbl>            <int>       <int>
##  1 Adelie  Torge…           39.1          18.7              181        3750
##  2 Adelie  Torge…           39.5          17.4              186        3800
##  3 Adelie  Torge…           40.3          18                195        3250
##  4 Adelie  Torge…           NA            NA                 NA          NA
##  5 Adelie  Torge…           36.7          19.3              193        3450
##  6 Adelie  Torge…           39.3          20.6              190        3650
##  7 Adelie  Torge…           38.9          17.8              181        3625
##  8 Adelie  Torge…           39.2          19.6              195        4675
##  9 Adelie  Torge…           34.1          18.1              193        3475
## 10 Adelie  Torge…           42            20.2              190        4250
## # … with 334 more rows, and 2 more variables: sex <fct>, year <int>
```


```r
penguins_dataset <- dataset(
  
  name = "penguins_dataset",
  
  initialize = function(df) {
    
    df <- na.omit(df) 
    
    # prepare input data (x)   
    # conveniently, the categorical data are already factors, so just need to convert to numeric
    # continuous data just stay as they are
    df$island <- as.numeric(df$island)
    df$sex <- as.numeric(df$sex)
    
    # everything but species goes into the input
    self$x <- as.matrix(df[ , -1]) %>% torch_tensor()
    
    # prepare target data (y)
    df$species <- as.numeric(df$species)
    # 
    self$y <- torch_tensor(df$species, dtype = torch_long())
    
  },
  
  .getitem = function(i) {
    
     list(x[i, ], y[i])
    
  },
  
  .length = function() {
    
    self$y$size()[[1]]
    
  }
 
)
```


```r
train_indices <- sample(1:nrow(penguins), 250)

train_ds <- penguins_dataset(penguins[train_indices, ])
```


