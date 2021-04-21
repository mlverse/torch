---
title: torch ecosystem
---

## Installation and use

-   Install torch running `install.packages("torch")`.

-   Run `library(torch)` to use it. Additional software will be downloaded and installed the first time you use torch.

## The torch ecosystem

There are a few extensions to the core torch package that are useful depending on the kind of data you are working on. See the list below:

{{< packages-hex >}}

## torch as a backend

torch can be used as a backend in various high-level modeling frameworks such as [tidymodels](http://tidymodels.org) and [fable](https://fable.tidyverts.org/). Here are some torch models and the contexts and frameworks in which they can be used:

| Package                                     | Framework  | Context                                                                                                |
|---------------------------------------------|------------|--------------------------------------------------------------------------------------------------------|
| [tabnet](https://github.com/mlverse/tabnet) | tidymodels | classification/regression/time series via ([modeltime](https://github.com/business-science/modeltime)) |
