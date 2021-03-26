---
title: torch in action
---

You've run your first [torch demo](/start/), and acquainted yourself with the [main actors](/technical/) (tensors, modules, optimizers)? Then you're ready to dive into applied examples. The list of examples keeps growing as the ecosystem evolves. What area are you interested in?

-   [Image recognition](#imag)
-   [Tabular data](#tab)
-   [Time series forecasting](#ts)

## Image recognition {\#tab}

-   [Bird classification](https://blogs.rstudio.com/ai/posts/2020-10-19-torch-image-classification/) is a multi-class classification task. In addition to being a blueprint for doing classification with torch, this introductory example shows how to load data, make use of pre-trained models, and benefit from learning rate schedulers.

-   [Brain image segmentation](https://blogs.rstudio.com/ai/posts/2020-11-30-torch-brain-segmentation/) builds a U-Net from scratch. This intermediate-level example is a great introduction to building your own modules, as well as custom datasets that perform data preprocessing and data augmentation for computer vision.

## Tabular data {\#tab}

-   [Labeling poisonous mushrooms](https://blogs.rstudio.com/ai/posts/2020-11-03-torch-tabular/) is a first introduction to handling a mix of numerical and categorical data, using embedding modules for the latter. It also provides a blueprint for creating torch models from scratch.

## Time series forecasting {\#ts}

-   [Introductory time-series forecasting with torch](https://blogs.rstudio.com/ai/posts/2021-03-10-forecasting-time-series-with-torch_1/) is a thorough introduction to RNNs (GRUs/LSTMs), explaining usage and terminology. [torch time series continued: A first go at multi-step prediction](https://blogs.rstudio.com/ai/posts/2021-03-11-forecasting-time-series-with-torch_2/) builds on this, and widens to the scope to multi-step-prediction.

-   [torch time series, take three: Sequence-to-sequence prediction](https://blogs.rstudio.com/ai/posts/2021-03-16-forecasting-time-series-with-torch_3/) and [torch time series, final episode: Attention](https://blogs.rstudio.com/ai/posts/2021-03-19-forecasting-time-series-with-torch_4/) expand on the prior two articles, introducing more advanced concepts like sequence-to-sequence processing and attention.

-   [Convolutional LSTM for spatial forecasting](https://blogs.rstudio.com/ai/posts/2020-12-17-torch-convlstm/) is an intermediate-level example that shows how to build a convolutional LSTM from scratch.
