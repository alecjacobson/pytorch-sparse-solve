#!/usr/bin/env python
import torch
from torch.autograd import gradcheck
#import torch_lu
import torch_chol

#solve = torch_lu.LinearDirectSolve.apply
solve = torch_chol.LinearDirectSolve.apply

# let us construct sparse tensor A(theta), and random vector b
# consider f = || A(theta)^-1 b ||^2

def f_sparse(A, b):
	return torch.linalg.norm(solve(A, b))**2

def f_dense(A, b):
	return torch.linalg.norm(torch.linalg.solve(A, b))**2

n = 10
theta_sparse = torch.rand(n, requires_grad=True, dtype=torch.double)
b_sparse = torch.rand(n, requires_grad=True, dtype=torch.double)
theta_dense = theta_sparse.detach().clone().requires_grad_(True)
b_dense = b_sparse.detach().clone().requires_grad_(True)

# let us define A: I + diag(cos(theta)) + [sin(sum(theta)) at anti-diagonal entries]:
A_dense = torch.diag(torch.cos(theta_dense)) + torch.sin(torch.sum(theta_dense))*torch.fliplr(torch.eye(n)) + torch.eye(n)

i = torch.hstack([torch.arange(n), torch.arange(n-1,-1,-1)])
j = torch.hstack([torch.arange(n), torch.arange(n)])
v = torch.hstack([1+torch.cos(theta_sparse), torch.sin(torch.sum(theta_sparse))*torch.ones(n)])
A_sparse = torch.sparse_coo_tensor(torch.stack([i, j]), v).coalesce()

if (torch.linalg.norm(A_dense - A_sparse.to_dense()) >= 1e-8):
	raise Exception("sparse and dense matrices do not match!")
#print(theta)
#print(A_dense)
#print(A_sparse.to_dense())
#print(b)

r_sparse = f_sparse(A_sparse, b_sparse)
r_dense = f_dense(A_dense, b_dense)

r_sparse.backward()
r_dense.backward()

print("gradient of theta_dense, theta_sparse:")
print('\t',theta_dense.grad)
print('\t',theta_sparse.grad)
print('difference between theta gradients: ', torch.linalg.norm(theta_dense.grad - theta_sparse.grad))
print()

print("gradient of b_dense, b_sparse:")
print('\t',b_dense.grad)
print('\t',b_sparse.grad)
print('difference between b gradients: ', torch.linalg.norm(b_dense.grad - b_sparse.grad))
print()
