---
title: "Iterative Bayesian optimization of a classification model"
tags: [tune, dials, parsnip, recipes, workflows]
categories: [model tuning]
type: learn-subsection
weight: 3
description: | 
  Identify the best hyperparameters for a model using Bayesian optimization of iterative search.
---


  


## Introduction

To use the code in this article, you will need to install the following packages: kernlab, modeldata, tidymodels, and tidyr.

Many of the examples for model tuning focus on [grid search](/learn/work/tune-svm/). For that method, all the candidate tuning parameter combinations are defined prior to evaluation. Alternatively, _iterative search_ can be used to analyze the existing tuning parameter results and then _predict_ which tuning parameters to try next. 

There are a variety of methods for iterative search and the focus in this article is on _Bayesian optimization_. For more information on this method, these resources might be helpful:

* [_Practical bayesian optimization of machine learning algorithms_](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C7&q=Practical+Bayesian+Optimization+of+Machine+Learning+Algorithms&btnG=) (2012). J Snoek, H Larochelle, and RP Adams. Advances in neural information.  

* [_A Tutorial on Bayesian Optimization for Machine Learning_](https://www.cs.toronto.edu/~rgrosse/courses/csc411_f18/tutorials/tut8_adams_slides.pdf) (2018). R Adams.

 * [_Gaussian Processes for Machine Learning_](http://www.gaussianprocess.org/gpml/) (2006). C E Rasmussen and C Williams.

* [Other articles!](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C7&q="Bayesian+Optimization"&btnG=)


## Cell segmenting revisited

To demonstrate this approach to tuning models, let's return to the cell segmentation data from the [Getting Started](/start/resampling/) article on resampling: 


```r
library(tidymodels)
library(modeldata)

# Load data
data(cells)

set.seed(2369)
tr_te_split <- initial_split(cells %>% select(-case), prop = 3/4)
cell_train <- training(tr_te_split)
cell_test  <- testing(tr_te_split)

set.seed(1697)
folds <- vfold_cv(cell_train, v = 10)
```

## The tuning scheme

Since the predictors are highly correlated, we can used a recipe to convert the original predictors to principal component scores. There is also slight class imbalance in these data; about 64% of the data are poorly segmented. To mitigate this, the data will be down-sampled at the end of the pre-processing so that the number of poorly and well segmented cells occur with equal frequency. We can use a recipe for all this pre-processing, but the number of principal components will need to be _tuned_ so that we have enough (but not too many) representations of the data. 


```r
cell_pre_proc <-
  recipe(class ~ ., data = cell_train) %>%
  step_YeoJohnson(all_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), num_comp = tune()) %>%
  step_downsample(class)
```

In this analysis, we will use a support vector machine to model the data. Let's use a radial basis function (RBF) kernel and tune its main parameter ($\sigma$). Additionally, the main SVM parameter, the cost value, also needs optimization. 


```r
svm_mod <-
  svm_rbf(mode = "classification", cost = tune(), rbf_sigma = tune()) %>%
  set_engine("kernlab")
```

These two objects (the recipe and model) will be combined into a single object via the `workflow()` function from the [workflows](https://tidymodels.github.io/workflows/) package; this object will be used in the optimization process. 


```r
svm_wflow <-
  workflow() %>%
  add_model(svm_mod) %>%
  add_recipe(cell_pre_proc)
```

From this object, we can derive information about what parameters are slated to be tuned. A parameter set is derived by: 


```r
svm_set <- parameters(svm_wflow)
svm_set
#> Collection of 3 parameters for tuning
#> 
#>         id parameter type object class
#>       cost           cost    nparam[+]
#>  rbf_sigma      rbf_sigma    nparam[+]
#>   num_comp       num_comp    nparam[+]
```

The default range for the number of PCA components is rather small for this data set. A member of the parameter set can be modified using the `update()` function. Let's constrain the search to one to twenty components by updating the `num_comp` parameter. Additionally, the lower bound of this parameter is set to zero which specifies that the original predictor set should also be evaluated (i.e., with no PCA step at all): 


```r
svm_set <- 
  svm_set %>% 
  update(num_comp = num_comp(c(0L, 20L)))
```

## Sequential tuning 

Bayesian optimization is a sequential method that uses a model to predict new candidate parameters for assessment. When scoring potential parameter value, the mean and variance of performance are predicted. The strategy used to define how these two statistical quantities are used is defined by an _acquisition function_. 

For example, one approach for scoring new candidates is to use a confidence bound. Suppose accuracy is being optimized. For a metric that we want to maximize, a lower confidence bound can be used. The multiplier on the standard error (denoted as `\(\kappa\)`) is a value that can be used to make trade-offs between **exploration** and **exploitation**. 

 * **Exploration** means that the search will consider candidates in untested space.

 * **Exploitation** focuses in areas where the previous best results occurred. 

The variance predicted by the Bayesian model is mostly spatial variation; the value will be large for candidate values that are not close to values that have already been evaluated. If the standard error multiplier is high, the search process will be more likely to avoid areas without candidate values in the vicinity. 

We'll use another acquisition function, _expected improvement_, that determines which candidates are likely to be helpful relative to the current best results. This is the default acquisition function. More information on these functions can be found in the [package vignette for acquisition functions](https://tidymodels.github.io/tune/articles/acquisition_functions.html). 


```r
set.seed(12)
search_res <-
  svm_wflow %>% 
  tune_bayes(
    resamples = folds,
    # To use non-default parameter ranges
    param_info = svm_set,
    # Generate five at semi-random to start
    initial = 5,
    iter = 50,
    # How to measure performance?
    metrics = metric_set(roc_auc),
    control = control_bayes(no_improve = 30, verbose = TRUE)
  )
#> 
#> >  Generating a set of 5 initial parameter results
#> ✓ Initialization complete
#> 
#> Optimizing roc_auc using the expected improvement
#> 
#> ── Iteration 1 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8655 (@iter 0)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.58, rbf_sigma=1.54e-09, num_comp=12
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8624 (+/-0.00897)
#> 
#> ── Iteration 2 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8655 (@iter 0)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.0251, rbf_sigma=6.36e-06, num_comp=16
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8606 (+/-0.00908)
#> 
#> ── Iteration 3 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8655 (@iter 0)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=23, rbf_sigma=1.02e-10, num_comp=7
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8634 (+/-0.00923)
#> 
#> ── Iteration 4 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8655 (@iter 0)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.0894, rbf_sigma=1.09e-10, num_comp=0
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8494 (+/-0.0116)
#> 
#> ── Iteration 5 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8655 (@iter 0)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.402, rbf_sigma=0.413, num_comp=20
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8236 (+/-0.00885)
#> 
#> ── Iteration 6 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8655 (@iter 0)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=24, rbf_sigma=0.942, num_comp=8
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8054 (+/-0.0114)
#> 
#> ── Iteration 7 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8655 (@iter 0)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=30.3, rbf_sigma=2.25e-06, num_comp=13
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8622 (+/-0.009)
#> 
#> ── Iteration 8 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8655 (@iter 0)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=25, rbf_sigma=1.07e-10, num_comp=20
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8655 (+/-0.00848)
#> 
#> ── Iteration 9 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8655 (@iter 0)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=2.1, rbf_sigma=5.29e-06, num_comp=0
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8494 (+/-0.0116)
#> 
#> ── Iteration 10 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8655 (@iter 0)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=9.87, rbf_sigma=0.000395, num_comp=20
#> i Estimating performance
#> ✓ Estimating performance
#> ♥ Newest results:	roc_auc=0.8681 (+/-0.00898)
#> 
#> ── Iteration 11 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8681 (@iter 10)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.073, rbf_sigma=0.000585, num_comp=0
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8509 (+/-0.0116)
#> 
#> ── Iteration 12 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8681 (@iter 10)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.00101, rbf_sigma=1.29e-07, num_comp=0
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8494 (+/-0.0116)
#> 
#> ── Iteration 13 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8681 (@iter 10)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.0553, rbf_sigma=0.000291, num_comp=12
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8625 (+/-0.00898)
#> 
#> ── Iteration 14 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8681 (@iter 10)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=11.8, rbf_sigma=0.00143, num_comp=20
#> i Estimating performance
#> ✓ Estimating performance
#> ♥ Newest results:	roc_auc=0.8691 (+/-0.00837)
#> 
#> ── Iteration 15 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8691 (@iter 14)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.0915, rbf_sigma=0.03, num_comp=20
#> i Estimating performance
#> ✓ Estimating performance
#> ♥ Newest results:	roc_auc=0.8728 (+/-0.00842)
#> 
#> ── Iteration 16 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8728 (@iter 15)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.0289, rbf_sigma=8.48e-09, num_comp=20
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8655 (+/-0.00848)
#> 
#> ── Iteration 17 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8728 (@iter 15)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.0021, rbf_sigma=0.0109, num_comp=20
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8696 (+/-0.00881)
#> 
#> ── Iteration 18 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8728 (@iter 15)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.461, rbf_sigma=0.908, num_comp=0
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.7732 (+/-0.0168)
#> 
#> ── Iteration 19 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8728 (@iter 15)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.00132, rbf_sigma=8.1e-08, num_comp=11
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8621 (+/-0.00933)
#> 
#> ── Iteration 20 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8728 (@iter 15)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=20.2, rbf_sigma=1.64e-09, num_comp=0
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8494 (+/-0.0116)
#> 
#> ── Iteration 21 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8728 (@iter 15)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.00173, rbf_sigma=0.126, num_comp=11
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8721 (+/-0.00749)
#> 
#> ── Iteration 22 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8728 (@iter 15)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.00853, rbf_sigma=0.989, num_comp=0
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.7369 (+/-0.0313)
#> 
#> ── Iteration 23 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8728 (@iter 15)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.00673, rbf_sigma=1.55e-10, num_comp=17
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.787 (+/-0.0485)
#> 
#> ── Iteration 24 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8728 (@iter 15)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.871, rbf_sigma=1.72e-10, num_comp=19
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.864 (+/-0.00842)
#> 
#> ── Iteration 25 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8728 (@iter 15)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=3.8, rbf_sigma=6.24e-10, num_comp=19
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.864 (+/-0.00842)
#> 
#> ── Iteration 26 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8728 (@iter 15)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=5.2, rbf_sigma=0.791, num_comp=1
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.7319 (+/-0.0173)
#> 
#> ── Iteration 27 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8728 (@iter 15)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.213, rbf_sigma=9.11e-10, num_comp=20
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8655 (+/-0.00848)
#> 
#> ── Iteration 28 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8728 (@iter 15)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=6.99, rbf_sigma=3.03e-10, num_comp=0
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8494 (+/-0.0116)
#> 
#> ── Iteration 29 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8728 (@iter 15)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.00102, rbf_sigma=0.344, num_comp=5
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8631 (+/-0.0079)
#> 
#> ── Iteration 30 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8728 (@iter 15)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=20.3, rbf_sigma=1.28e-05, num_comp=3
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8503 (+/-0.0112)
#> 
#> ── Iteration 31 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8728 (@iter 15)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.0012, rbf_sigma=3.75e-05, num_comp=7
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8634 (+/-0.00923)
#> 
#> ── Iteration 32 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8728 (@iter 15)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.0142, rbf_sigma=2.58e-10, num_comp=1
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.7015 (+/-0.0374)
#> 
#> ── Iteration 33 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8728 (@iter 15)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.00411, rbf_sigma=0.557, num_comp=1
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.747 (+/-0.0173)
#> 
#> ── Iteration 34 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8728 (@iter 15)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.161, rbf_sigma=0.167, num_comp=1
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.7541 (+/-0.0177)
#> 
#> ── Iteration 35 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8728 (@iter 15)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=2.48, rbf_sigma=0.783, num_comp=19
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.7748 (+/-0.014)
#> 
#> ── Iteration 36 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8728 (@iter 15)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.0138, rbf_sigma=0.624, num_comp=19
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.7938 (+/-0.0117)
#> 
#> ── Iteration 37 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8728 (@iter 15)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.00341, rbf_sigma=1.11e-10, num_comp=2
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.7311 (+/-0.0404)
#> 
#> ── Iteration 38 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8728 (@iter 15)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.00113, rbf_sigma=1.48e-10, num_comp=14
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.7888 (+/-0.0489)
#> 
#> ── Iteration 39 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8728 (@iter 15)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=17.1, rbf_sigma=1.71e-10, num_comp=9
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8638 (+/-0.00874)
#> 
#> ── Iteration 40 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8728 (@iter 15)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=13.3, rbf_sigma=0.968, num_comp=17
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.7691 (+/-0.0158)
#> 
#> ── Iteration 41 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8728 (@iter 15)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.0026, rbf_sigma=0.995, num_comp=3
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8496 (+/-0.0093)
#> 
#> ── Iteration 42 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8728 (@iter 15)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=23.6, rbf_sigma=0.856, num_comp=13
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.7972 (+/-0.0144)
#> 
#> ── Iteration 43 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8728 (@iter 15)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.00142, rbf_sigma=7.1e-06, num_comp=18
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8593 (+/-0.00882)
#> 
#> ── Iteration 44 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8728 (@iter 15)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=31.4, rbf_sigma=1.7e-10, num_comp=15
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8616 (+/-0.00899)
#> 
#> ── Iteration 45 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8728 (@iter 15)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=31.4, rbf_sigma=0.0118, num_comp=6
#> i Estimating performance
#> ✓ Estimating performance
#> ♥ Newest results:	roc_auc=0.8779 (+/-0.00726)
#> 
#> ── Iteration 46 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8779 (@iter 45)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=11, rbf_sigma=0.718, num_comp=10
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8247 (+/-0.0103)
#> 
#> ── Iteration 47 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8779 (@iter 45)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=27.1, rbf_sigma=3.61e-06, num_comp=8
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8645 (+/-0.00874)
#> 
#> ── Iteration 48 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8779 (@iter 45)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=20.4, rbf_sigma=1.23e-10, num_comp=4
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8513 (+/-0.0109)
#> 
#> ── Iteration 49 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8779 (@iter 45)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.0011, rbf_sigma=0.677, num_comp=16
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8075 (+/-0.0119)
#> 
#> ── Iteration 50 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#> 
#> i Current best:		roc_auc=0.8779 (@iter 45)
#> i Gaussian process model
#> ✓ Gaussian process model
#> i Generating 5000 candidates
#> i Predicted candidates
#> i cost=0.00133, rbf_sigma=0.592, num_comp=14
#> i Estimating performance
#> ✓ Estimating performance
#> ⓧ Newest results:	roc_auc=0.8311 (+/-0.014)
```

The resulting tibble is a stacked set of rows of the rsample object with an additional column for the iteration number:


```r
search_res
#> #  10-fold cross-validation 
#> # A tibble: 510 x 5
#>    splits             id     .metrics         .notes           .iter
#>  * <list>             <chr>  <list>           <list>           <dbl>
#>  1 <split [1.4K/152]> Fold01 <tibble [5 × 6]> <tibble [0 × 1]>     0
#>  2 <split [1.4K/152]> Fold02 <tibble [5 × 6]> <tibble [0 × 1]>     0
#>  3 <split [1.4K/152]> Fold03 <tibble [5 × 6]> <tibble [0 × 1]>     0
#>  4 <split [1.4K/152]> Fold04 <tibble [5 × 6]> <tibble [0 × 1]>     0
#>  5 <split [1.4K/152]> Fold05 <tibble [5 × 6]> <tibble [0 × 1]>     0
#>  6 <split [1.4K/151]> Fold06 <tibble [5 × 6]> <tibble [0 × 1]>     0
#>  7 <split [1.4K/151]> Fold07 <tibble [5 × 6]> <tibble [0 × 1]>     0
#>  8 <split [1.4K/151]> Fold08 <tibble [5 × 6]> <tibble [0 × 1]>     0
#>  9 <split [1.4K/151]> Fold09 <tibble [5 × 6]> <tibble [0 × 1]>     0
#> 10 <split [1.4K/151]> Fold10 <tibble [5 × 6]> <tibble [0 × 1]>     0
#> # … with 500 more rows
```

As with grid search, we can summarize the results over resamples:


```r
estimates <- 
  collect_metrics(search_res) %>% 
  arrange(.iter)

estimates
#> # A tibble: 55 x 9
#>        cost rbf_sigma num_comp .iter .metric .estimator  mean     n std_err
#>       <dbl>     <dbl>    <int> <dbl> <chr>   <chr>      <dbl> <int>   <dbl>
#>  1  0.00207  1.56e- 5       10     0 roc_auc binary     0.864    10 0.00888
#>  2  0.0304   6.41e- 9        5     0 roc_auc binary     0.859    10 0.00922
#>  3  0.348    4.43e- 2        1     0 roc_auc binary     0.757    10 0.0177 
#>  4  1.45     2.04e- 3       15     0 roc_auc binary     0.865    10 0.00962
#>  5 15.5      1.28e- 7       20     0 roc_auc binary     0.865    10 0.00848
#>  6  0.580    1.54e- 9       12     1 roc_auc binary     0.862    10 0.00897
#>  7  0.0251   6.36e- 6       16     2 roc_auc binary     0.861    10 0.00908
#>  8 23.0      1.02e-10        7     3 roc_auc binary     0.863    10 0.00923
#>  9  0.0894   1.09e-10        0     4 roc_auc binary     0.849    10 0.0116 
#> 10  0.402    4.13e- 1       20     5 roc_auc binary     0.824    10 0.00885
#> # … with 45 more rows
```


The best performance of the initial set of candidate values was `AUC = 0.865 `. The best results were achieved at iteration 45 with a corresponding AUC value of 0.878. The five best results are:


```r
show_best(search_res, metric = "roc_auc")
#> # A tibble: 5 x 9
#>       cost rbf_sigma num_comp .iter .metric .estimator  mean     n std_err
#>      <dbl>     <dbl>    <int> <dbl> <chr>   <chr>      <dbl> <int>   <dbl>
#> 1 31.4       0.0118         6    45 roc_auc binary     0.878    10 0.00726
#> 2  0.0915    0.0300        20    15 roc_auc binary     0.873    10 0.00842
#> 3  0.00173   0.126         11    21 roc_auc binary     0.872    10 0.00749
#> 4  0.00210   0.0109        20    17 roc_auc binary     0.870    10 0.00881
#> 5 11.8       0.00143       20    14 roc_auc binary     0.869    10 0.00837
```

A plot of the search iterations can be created via:


```r
autoplot(search_res, type = "performance")
```

<img src="figs/bo-plot-1.svg" width="672" />

There are many parameter combinations have roughly equivalent results. 

How did the parameters change over iterations? Since two of the parameters are usually treated on the log scale, we can use `mutate()` to transform them, and then construct a plot using ggplot2:  



```r
library(tidyr)

collect_metrics(search_res) %>%
  select(-.metric,-.estimator,-mean,-n,-std_err) %>%
  mutate(cost = log10(cost), 
         rbf_sigma = log10(rbf_sigma)) %>%
  pivot_longer(cols = c(-.iter),
               names_to = "parameter",
               values_to = "value") %>%
  ggplot(aes(x = .iter, y = value)) +
  geom_point() +
  facet_wrap( ~ parameter, scales = "free_y")
```

<img src="figs/bo-param-plot-1.svg" width="672" />




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
#>  kernlab    * 0.9-29  2019-11-12 [1] CRAN (R 3.6.0)
#>  modeldata  * 0.0.1   2019-12-06 [1] CRAN (R 3.6.0)
#>  parsnip    * 0.1.0   2020-04-09 [1] CRAN (R 3.6.2)
#>  purrr      * 0.3.3   2019-10-18 [1] CRAN (R 3.6.0)
#>  recipes    * 0.1.10  2020-03-18 [1] CRAN (R 3.6.0)
#>  rlang      * 0.4.5   2020-03-01 [1] CRAN (R 3.6.0)
#>  rsample    * 0.0.6   2020-03-31 [1] CRAN (R 3.6.2)
#>  tibble     * 2.1.3   2019-06-06 [1] CRAN (R 3.6.2)
#>  tidymodels * 0.1.0   2020-02-16 [1] CRAN (R 3.6.0)
#>  tidyr      * 1.0.2   2020-01-24 [1] CRAN (R 3.6.0)
#>  tune       * 0.1.0   2020-04-02 [1] CRAN (R 3.6.2)
#>  workflows  * 0.1.1   2020-03-17 [1] CRAN (R 3.6.0)
#>  yardstick  * 0.0.6   2020-03-17 [1] CRAN (R 3.6.0)
#> 
#> [1] /Library/Frameworks/R.framework/Versions/3.6/Resources/library
```
 
