---
title: torch in action
---

You've run your first [torch demo](/start/), and acquainted yourself with the [main actors](/technical/) (tensors, modules, optimizers)? Then you're ready to dive into applied examples. The list of examples keeps growing as the ecosystem evolves. What area are you interested in?

-   [Image recognition](#imag)
-   [Tabular data](#tab)
-   [Time series forecasting](#ts)
-   [Audio processing](#aud)

## Image recognition {#imag}

-   A thorough introduction to the [why and how of image processing](https://skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/image_classification_1.html) with deep learning is found in our book, [Deep Learning and Scientific Computing with R torch](https://skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/image_classification_1.html).

```{=html}
<!-- -->
```
-   [Bird classification](https://blogs.rstudio.com/ai/posts/2020-10-19-torch-image-classification/) is a multi-class classification task. In addition to being a blueprint for doing classification with torch, this introductory example shows how to load data, make use of pre-trained models, and benefit from learning rate schedulers.

-   [Brain image segmentation](https://blogs.rstudio.com/ai/posts/2020-11-30-torch-brain-segmentation/) builds a U-Net from scratch. This intermediate-level example is a great introduction to building your own modules, as well as custom datasets that perform data preprocessing and data augmentation for computer vision.

## Tabular data {#tab}

-   An interesting use case that illustrates the importance of domain knowledge is discussed the `torch` [book](https://skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/tabular_data.html).

```{=html}
<!-- -->
```
-   [Labeling poisonous mushrooms](https://blogs.rstudio.com/ai/posts/2020-11-03-torch-tabular/) is a first introduction to handling a mix of numerical and categorical data, using embedding modules for the latter. It also provides a blueprint for creating torch models from scratch.

-   [torch, tidymodels, and high-energy physics](https://blogs.rstudio.com/ai/posts/2021-02-11-tabnet/) introduces `tabnet`, a torch implementation of "TabNet: Attentive Interpretable Tabular Learning" that is fully integrated with the `tidymodels` framework. Thanks to `tidymodels` integration, both pre-processing and hyperparameter tuning need a minimal amount of code.

## Time series forecasting {#ts}

-   The general ideas behind time-series prediction with deep learning are discussed in-depth in the book, [Deep Learning and Scientific Computing with R `torch`](https://skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/time_series.html).

```{=html}
<!-- -->
```
-   [Introductory time-series forecasting with torch](https://blogs.rstudio.com/ai/posts/2021-03-10-forecasting-time-series-with-torch_1/) is a thorough introduction to RNNs (GRUs/LSTMs), explaining usage and terminology. [torch time series continued: A first go at multi-step prediction](https://blogs.rstudio.com/ai/posts/2021-03-11-forecasting-time-series-with-torch_2/) builds on this, and widens to the scope to multi-step-prediction.

-   [torch time series, take three: Sequence-to-sequence prediction](https://blogs.rstudio.com/ai/posts/2021-03-16-forecasting-time-series-with-torch_3/) and [torch time series, final episode: Attention](https://blogs.rstudio.com/ai/posts/2021-03-19-forecasting-time-series-with-torch_4/) expand on the prior two articles, introducing more advanced concepts like sequence-to-sequence processing and attention.

-   [Convolutional LSTM for spatial forecasting](https://blogs.rstudio.com/ai/posts/2020-12-17-torch-convlstm/) is an intermediate-level example that shows how to build a convolutional LSTM from scratch.

## Audio processing {#aud}

-   In its chapter on audio classification, the [`torch` book](https://skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/audio_classification.html) shows, by example, the usefulness of integrating Fourier-domain representations with deep learning.
