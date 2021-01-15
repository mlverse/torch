---
title: "What if? Experiments and adaptations"
weight: 2
description: | 
  Modify and experiment with the guess-the-correlation model.
---

What is it that we've done in the previous tutorial? Put abstractly, we've trained a network to *take in images* and *output* a *continuous* numerical value.

In the process, we've made decisions all the time -- what, and how many, layers to use; how to calculate the loss; what optimization algorithm to apply; how long to train; and more. We can't go into all of them here, and we can't go into great detail. But the good thing is: With deep learning, you can always experiment and find out. (In fact, more often than not, experiment and find out is the only way to find out!)

So this page is basically an invitation to try out things for yourself.

# What if ... we were working with a different kind of data -- not images?

With deep learning, the type of input data decides the type of architecture we use. Or architectures. (Quick note: By architecture, I mean something more like a family than a specific model. For example, convolutional neural networks (CNNs) would be one; or Long Short-Term Memory model (LSTM); or Transformer.)

Sometimes there are several established architectures for a problem; sometimes there's one most prominent family. Even in the latter case though, there is no rule you *have* to use it.

For example, take our scatterplot images. The canonical architecture in image recognition are CNNs. *But*, you could still work on image data using nothing but linear layers. Depending on the task, this may or may not work so well.

So why not give it a try? If you want to try this, there are two places you have to modify:

-   One,
