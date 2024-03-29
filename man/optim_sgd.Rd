% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/optim-sgd.R
\name{optim_sgd}
\alias{optim_sgd}
\title{SGD optimizer}
\usage{
optim_sgd(
  params,
  lr = optim_required(),
  momentum = 0,
  dampening = 0,
  weight_decay = 0,
  nesterov = FALSE
)
}
\arguments{
\item{params}{(iterable): iterable of parameters to optimize or dicts defining
parameter groups}

\item{lr}{(float): learning rate}

\item{momentum}{(float, optional): momentum factor (default: 0)}

\item{dampening}{(float, optional): dampening for momentum (default: 0)}

\item{weight_decay}{(float, optional): weight decay (L2 penalty) (default: 0)}

\item{nesterov}{(bool, optional): enables Nesterov momentum (default: FALSE)}
}
\description{
Implements stochastic gradient descent (optionally with momentum).
Nesterov momentum is based on the formula from
On the importance of initialization and momentum in deep learning.
}
\section{Note}{


The implementation of SGD with Momentum-Nesterov subtly differs from
Sutskever et. al. and implementations in some other frameworks.

Considering the specific case of Momentum, the update can be written as
\deqn{
  \begin{array}{ll}
v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
p_{t+1} & = p_{t} - \mbox{lr} * v_{t+1},
\end{array}
}

where \eqn{p}, \eqn{g}, \eqn{v} and \eqn{\mu} denote the
parameters, gradient, velocity, and momentum respectively.

This is in contrast to Sutskever et. al. and
other frameworks which employ an update of the form

\deqn{
  \begin{array}{ll}
v_{t+1} & = \mu * v_{t} + \mbox{lr} * g_{t+1}, \\
p_{t+1} & = p_{t} - v_{t+1}.
\end{array}
}
The Nesterov version is analogously modified.
}

\section{Warning}{
If you need to move a model to GPU via \verb{$cuda()}, please do so before
constructing optimizers for it. Parameters of a model after \verb{$cuda()}
will be different objects from those before the call. In general, you
should make sure that the objects pointed to by model parameters subject
to optimization remain the same over the whole lifecycle of optimizer
creation and usage.
}

\examples{
if (torch_is_installed()) {
\dontrun{
optimizer <- optim_sgd(model$parameters(), lr = 0.1, momentum = 0.9)
optimizer$zero_grad()
loss_fn(model(input), target)$backward()
optimizer$step()
}

}
}
