---
title: "Classification models using a neural network"
tags: [rsample, parsnip]
categories: [model fitting]
type: learn-subsection
weight: 2
description: | 
  Train a classification model and evaluate its performance.
---


  



## Introduction

To use the code in this article, you will need to install the following packages: keras and tidymodels. You will also need the python keras library installed (see `?keras::install_keras()`).

We can create classification models with the tidymodels package [parsnip](https://tidymodels.github.io/parsnip/) to predict categorical quantities or class labels. Here, let's fit a single classification model using a neural network and evaluate using a validation set. While the [tune](https://tidymodels.github.io/tune/) package has functionality to also do this, the parsnip package is the center of attention in this article so that we can better understand its usage. 

## Fitting a neural network


Let's fit a model to a small, two predictor classification data set. The data are in the modeldata package (part of tidymodels) and have been split into training, validation, and test data sets. In this analysis, the test set is left untouched; this article tries to emulate a good data usage methodology where the test set would only be evaluated once at the end after a variety of models have been considered. 



```r
data(bivariate)
nrow(bivariate_train)
#> [1] 1009
nrow(bivariate_val)
#> [1] 300
```

A plot of the data shows two right-skewed predictors: 


```r
ggplot(bivariate_train, aes(x = A, y = B, col = Class)) + 
  geom_point(alpha = .2)
```

<img src="figs/biv-plot-1.svg" width="576" />

Let's use a single hidden layer neural network to predict the outcome. To do this, we transform the predictor columns to be more symmetric (via the `step_BoxCox()` function) and on a common scale (using `step_normalize()`). We can use [recipes](https://tidymodels.github.io/recipes/) to do so:


```r
biv_rec <- 
  recipe(Class ~ ., data = bivariate_train) %>%
  step_BoxCox(all_predictors())%>%
  step_normalize(all_predictors()) %>%
  prep(training = bivariate_train, retain = TRUE)

# We will juice() to get the processed training set back

# For validation:
val_normalized <- bake(biv_rec, new_data = bivariate_val, all_predictors())
# For testing when we arrive at a final model: 
test_normalized <- bake(biv_rec, new_data = bivariate_test, all_predictors())
```

We can use the keras package to fit a model with 5 hidden units and a 10% dropout rate, to regularize the model:


```r
set.seed(57974)
nnet_fit <-
  mlp(epochs = 100, hidden_units = 5, dropout = 0.1) %>%
  set_mode("classification") %>% 
  # Also set engine-specific `verbose` argument to prevent logging the results: 
  set_engine("keras", verbose = 0) %>%
  fit(Class ~ ., data = juice(biv_rec))

nnet_fit
#> parsnip model object
#> 
#> Fit time:  8.7s 
#> Model
#> Model: "sequential"
#> ________________________________________________________________________________
#> Layer (type)                        Output Shape                    Param #     
#> ================================================================================
#> dense (Dense)                       (None, 5)                       15          
#> ________________________________________________________________________________
#> dense_1 (Dense)                     (None, 5)                       30          
#> ________________________________________________________________________________
#> dropout (Dropout)                   (None, 5)                       0           
#> ________________________________________________________________________________
#> dense_2 (Dense)                     (None, 2)                       12          
#> ================================================================================
#> Total params: 57
#> Trainable params: 57
#> Non-trainable params: 0
#> ________________________________________________________________________________
```

## Model performance

In parsnip, the `predict()` function can be used to characterize performance on the validation set. Since parsnip always produces tibble outputs, these can just be column bound to the original data: 


```r
val_results <- 
  bivariate_val %>%
  bind_cols(
    predict(nnet_fit, new_data = val_normalized),
    predict(nnet_fit, new_data = val_normalized, type = "prob")
  )
val_results %>% slice(1:5)
#> # A tibble: 5 x 6
#>       A     B Class .pred_class .pred_One .pred_Two
#>   <dbl> <dbl> <fct> <fct>           <dbl>     <dbl>
#> 1 1061.  74.5 One   Two             0.473    0.527 
#> 2 1241.  83.4 One   Two             0.484    0.516 
#> 3  939.  71.9 One   One             0.636    0.364 
#> 4  813.  77.1 One   One             0.925    0.0746
#> 5 1706.  92.8 Two   Two             0.355    0.645

val_results %>% roc_auc(truth = Class, .pred_One)
#> # A tibble: 1 x 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 roc_auc binary         0.815

val_results %>% accuracy(truth = Class, .pred_class)
#> # A tibble: 1 x 3
#>   .metric  .estimator .estimate
#>   <chr>    <chr>          <dbl>
#> 1 accuracy binary         0.737

val_results %>% conf_mat(truth = Class, .pred_class)
#>           Truth
#> Prediction One Two
#>        One 150  27
#>        Two  52  71
```

Let's also create a grid to get a visual sense of the class boundary for the validation set.


```r
a_rng <- range(bivariate_train$A)
b_rng <- range(bivariate_train$B)
x_grid <-
  expand.grid(A = seq(a_rng[1], a_rng[2], length.out = 100),
              B = seq(b_rng[1], b_rng[2], length.out = 100))
x_grid_trans <- bake(biv_rec, x_grid)

# Make predictions using the transformed predictors but 
# attach them to the predictors in the original units: 
x_grid <- 
  x_grid %>% 
  bind_cols(predict(nnet_fit, x_grid_trans, type = "prob"))

ggplot(x_grid, aes(x = A, y = B)) + 
  geom_contour(aes(z = .pred_One), breaks = .5, col = "black") + 
  geom_point(data = bivariate_val, aes(col = Class), alpha = 0.3)
```

<img src="figs/biv-boundary-1.svg" width="576" />



## Session information


```
#> ─ Session info ───────────────────────────────────────────────────────────────
#>  setting  value                       
#>  version  R version 3.6.2 (2019-12-12)
#>  os       macOS Mojave 10.14.6        
#>  system   x86_64, darwin15.6.0        
#>  ui       X11                         
#>  language (EN)                        
#>  collate  en_US.UTF-8                 
#>  ctype    en_US.UTF-8                 
#>  tz       America/Denver              
#>  date     2020-04-17                  
#> 
#> ─ Packages ───────────────────────────────────────────────────────────────────
#>  package    * version date       lib source        
#>  broom      * 0.5.5   2020-02-29 [1] CRAN (R 3.6.0)
#>  dials      * 0.0.6   2020-04-03 [1] CRAN (R 3.6.2)
#>  dplyr      * 0.8.5   2020-03-07 [1] CRAN (R 3.6.0)
#>  ggplot2    * 3.3.0   2020-03-05 [1] CRAN (R 3.6.0)
#>  infer      * 0.5.1   2019-11-19 [1] CRAN (R 3.6.0)
#>  keras        2.2.5.0 2019-10-08 [1] CRAN (R 3.6.0)
#>  parsnip    * 0.1.0   2020-04-09 [1] CRAN (R 3.6.2)
#>  purrr      * 0.3.3   2019-10-18 [1] CRAN (R 3.6.0)
#>  recipes    * 0.1.10  2020-03-18 [1] CRAN (R 3.6.0)
#>  rlang        0.4.5   2020-03-01 [1] CRAN (R 3.6.0)
#>  rsample    * 0.0.6   2020-03-31 [1] CRAN (R 3.6.2)
#>  tibble     * 2.1.3   2019-06-06 [1] CRAN (R 3.6.2)
#>  tidymodels * 0.1.0   2020-02-16 [1] CRAN (R 3.6.0)
#>  tune       * 0.1.0   2020-04-02 [1] CRAN (R 3.6.2)
#>  workflows  * 0.1.1   2020-03-17 [1] CRAN (R 3.6.0)
#>  yardstick  * 0.0.6   2020-03-17 [1] CRAN (R 3.6.0)
#> 
#> [1] /Library/Frameworks/R.framework/Versions/3.6/Resources/library
```
