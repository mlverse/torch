# Lstsq

Lstsq

## Arguments

- self:

  (Tensor) the matrix \\B\\

- A:

  (Tensor) the \\m\\ by \\n\\ matrix \\A\\

## Note

    The case when \eqn{m < n} is not supported on the GPU.

## lstsq(input, A, out=NULL) -\> Tensor

Computes the solution to the least squares and least norm problems for a
full rank matrix \\A\\ of size \\(m \times n)\\ and a matrix \\B\\ of
size \\(m \times k)\\.

If \\m \geq n\\, `torch_lstsq()` solves the least-squares problem:

\$\$ \begin{array}{ll} \min_X & \\AX-B\\\_2. \end{array} \$\$ If \\m \<
n\\, `torch_lstsq()` solves the least-norm problem:

\$\$ \begin{array}{llll} \min_X & \\X\\\_2 & \mbox{subject to} & AX = B.
\end{array} \$\$ Returned tensor \\X\\ has shape \\(\mbox{max}(m, n)
\times k)\\. The first \\n\\ rows of \\X\\ contains the solution. If \\m
\geq n\\, the residual sum of squares for the solution in each column is
given by the sum of squares of elements in the remaining \\m - n\\ rows
of that column.
