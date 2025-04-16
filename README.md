# PyTorch Sparse Solve

This small library provides a python class `SparseSolve` to use pytorch to
back-propagate through function's which involve solving against a sparse matrix
whose entries are differentiation variables. For example, consider that we have
a function $\mathcal{L} : \mathbb{R}^d \rightarrow \mathbb{R}$ where $\mathcal{L}$ is defined as:

$$
\mathcal{L}(w) = \frac{1}{2} \left\| A(w)^{-1} b \right\|^2
$$

and $A: \mathbb{R}^d \rightarrow \mathbb{R}^{n \times n}$ is some
*sparse*-matrix function of the model parameters $w$.
Correspondingly in pytorch we might write:

```python
b = torch.ones(n, dtype=torch.double)
# x = A⁻¹ b
x = torch.linalg.solve(A,b)
# L(w) = ½‖x‖² = ½‖A(w)⁻¹ b‖²
L = torch.sum(x**2) / 2
```

For example, suppose that $A$ is defined to be the identity matrix plus the weighted graph
Laplacian for a sparse set of $d$ edges. In mathematical terms, we can write:

$$
A_{ij} = \begin{cases}
-w_e & \text{if } (i,j) \text{ or } (j,i) \text{ is the $e$-th edge} \\
1 + \sum\limits_{k\neq i} A_{ik} & \text{if } i = j \\
0 & \text{otherwise}
\end{cases}.
$$

Unfortunately, pytorch doesn't support sparse matrices well by default. So if we
were to build a $A$ matrix, we would have to build a dense matrix. For example,
assuming a sparse set of random edges, we might write something like:

```python
# size of problem
n = 10
# seed
torch.manual_seed(0)
# E is a #E by 2 list of edges (i,j) ∈ [0,n)²
E = torch.unique(torch.randint(0, n, (n*6, 2), dtype=torch.int64), dim=0)
# w is a #E vector of parameters
w = torch.ones(E.shape[0], dtype=torch.double, requires_grad=True)
# A = I + WeightedGraphLaplacian(E,w)
diag = torch.arange(n)
indices = torch.stack([torch.cat([diag,E[:,0],E[:,1],E[:,0],E[:,1]]),torch.cat([diag,E[:,1],E[:,0],E[:,0],E[:,1]])])
values = torch.cat([torch.ones(n, dtype=torch.double), -w, -w, w, w])

# Build A as dense matrix (default for torch)
A = torch.zeros(n, n, dtype=torch.double)
A.index_put_((indices[0], indices[1]), values, accumulate=True)
```

The forward pass is of course $O(n^2)$ just to construct `A`, but calling
`torch.linalg.solve(A, b)` is $O(n^3)$. The backward pass is similarly $O(n^3)$:

```python
# very slow for large n
L.backward()
dLdw = w.grad.clone().detach()
```

This default dense pytorch code will choke as $n$ increases. 

Fortunately instead, we can use `torch.sparse_coo_tensor` and `SparseSolve` to
construct and solve against $A$ in a sparse way while maintaining
differentiability. 

```python
import sparse_solver
# build A as a torch.sparse_coo_tensor 
A_sparse = torch.sparse_coo_tensor(indices, values, size=(n, n), dtype=torch.double).coalesce()
# x = A⁻¹ b
x = sparse_solver.SparseSolver.apply(A_sparse, b)
# L(w) = ½‖x‖² = ½‖A(w)⁻¹ b‖²
L = torch.sum(x**2) / 2
w.grad = None
# Efficient even for large n
L.backward()
dLdw_sparse = w.grad.clone().detach()
```

`SparseSolve` uses CPU-based sparse Cholesky factorization and GPU-back/forward-substitution in the forward pass and cache the factorization for efficient GPU backward pass. The precise asymptotic behavior depends on the sparsity pattern and ability to permute the matrix well, but for common patterns it will be something like $O(n^p)$ where $1\leq p \leq 2$.

For examples like the one above, as $n$ increases torch indeed measures performance looking something like $n^{2.5}$
and SparseSolve measures performance very close to $n^{1.0}$.

![](benchmark.png)

## Use

Install with pip:

    python -m pip install . 

Run tests

    pytest

Run minimal example above

    python examples/minimal.py

Run benchmark

    python examples/benchmark.py


## To-do list

 - [ ] Add fuller example (e.g., "Fast Quasi-Harmonic Weights for Geometric Data Interpolation", or inverse design of mass-spring cantilever)

You might also be interested in https://github.com/alecjacobson/indexed_sum

https://github.com/flaport/torch_sparse_solve appears to be similar, but supports batching and uses LU instead of Cholesky.

_Original code from Aravind Ramakrishnan._
