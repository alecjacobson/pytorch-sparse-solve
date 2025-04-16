import torch
from cholespy import CholeskySolverD, MatrixType

class SparseSolver(torch.autograd.Function):
    CHOL = None # cholesky decomposition
    A0 = torch.sparse_coo_tensor((1,1)).coalesce() # current matrix corresponding to CHOL

    @staticmethod
    def forward(ctx, A, b):
        if A.layout != torch.sparse_coo:
            A = A.to_sparse_coo()

        if not (torch.equal(SparseSolver.A0.indices(), A.indices()) and torch.equal(SparseSolver.A0.values(), A.values())):
        # don't factor matrix unless necessary
            SparseSolver.A0 = A

            ind = SparseSolver.A0.indices()
            rows = ind[0,:]
            cols = ind[1,:]
            vals = SparseSolver.A0.values()

            SparseSolver.CHOL = CholeskySolverD(SparseSolver.A0.size(0), rows, cols, vals, MatrixType.COO)

        x = torch.zeros_like(b, dtype=torch.double)
        SparseSolver.CHOL.solve(b, x)
        ctx.save_for_backward(b, SparseSolver.A0.indices(), x)
        return x

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        b, ind, res = ctx.saved_tensors
        grad_A = grad_b = None

        if ctx.needs_input_grad[0]:
        # partial1 = −diag(inv(A⊙S)^⊤⋅g)⋅S⋅diag(inv(A⊙S)⋅b) # but for symmetric A, S, we can drop the transposes
            n = b.size(0)
            S = torch.sparse_coo_tensor(ind, torch.ones(ind.size(1)), (n,n), dtype=torch.double)

            p1left = torch.zeros_like(b, dtype=torch.double)
            SparseSolver.CHOL.solve(grad_output.clone().detach().double(), p1left)
            p1left = torch.sparse_coo_tensor(torch.stack([torch.arange(n), torch.arange(n)], 0), p1left)

            p1right = torch.sparse_coo_tensor(torch.stack([torch.arange(n), torch.arange(n)], 0), res, dtype=torch.double)

            grad_A = -p1left @ S @ p1right

        if ctx.needs_input_grad[1]:
        # partial2 = inv(A⊙S)^⊤⋅g # drop transpose bc symmetric
            grad_b = torch.zeros_like(b, dtype=torch.double)
            SparseSolver.CHOL.solve(grad_output.clone().detach().double(), grad_b)

        return grad_A, grad_b
