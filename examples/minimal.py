import torch

# size of problem
n = 10
b = torch.ones(n, dtype=torch.double)
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

# x = A⁻¹ b
x = torch.linalg.solve(A,b)
# L(w) = ½‖x‖² = ½‖A(w)⁻¹ b‖²
L = torch.sum(x**2) / 2
w.grad = None
# very slow for large n
L.backward()
dLdw = w.grad.clone().detach()

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

print("‖dLdw - dLdw_sparse‖ = ", torch.linalg.norm(dLdw - dLdw_sparse).item())
