import sparse_solver
import torch
import time

import matplotlib.pyplot as plt
import numpy as np


# is torch_sparse_solve available?
try:
    import torch_sparse_solve
    torch_sparse_solve_available = True
except ImportError:
    torch_sparse_solve_available = False

print(f"torch_sparse_solve available: {torch_sparse_solve_available}")

def x_dense(theta,b):
    n = b.shape[0]
    # A is defined so that:
    # ½ x^T A x = ½ x^T I x + ∑_i theta_i (x_i - x_{i+1})^2
    #
    # A = 0
    # A(i,i) += 1 + theta(i) + theta(i+1) 
    # A(i,j) += -theta(i) if j == i+1 or i == j+1
    
    # Start with identity
    A = torch.eye(n, dtype=torch.double)
    
    # Compute indices for cyclic neighbor (i, i+1 mod n)
    i = torch.arange(n)
    j = (i + 1) % n
    
    # Add theta[i] to (i, j) and (j, i)
    A[i, j] -= theta
    A[j, i] -= theta
    
    i = torch.arange(n)
    j = (i + 1) % n
    A[i,i] += theta
    A[j,j] += theta
    
    
    x = torch.linalg.solve(A,b)
    return x

def A_sparse(theta,n):
    # Identity part: (i, i) → 1
    diag_i = torch.arange(n)
    diag_j = torch.arange(n)
    diag_val = torch.ones(n, dtype=torch.double) + theta + theta.roll(1)

    # Off-diagonal part: (i, i+1) and (i+1, i) → -theta[i]
    i = torch.arange(n)
    j = (i + 1) % n

    # Stack (i,j) and (j,i)
    off_i = torch.cat([i, j])
    off_j = torch.cat([j, i])
    off_val = torch.cat([-theta, -theta])

    # Combine all
    indices = torch.stack([torch.cat([diag_i, off_i]),
                           torch.cat([diag_j, off_j])])
    values = torch.cat([diag_val, off_val])

    A = torch.sparse_coo_tensor(indices, values, size=(n, n), dtype=torch.double).coalesce()
    return A

def x_sparse(theta,b):
    n = b.shape[0]
    A = A_sparse(theta, n)
    x = sparse_solver.SparseSolver.apply(A, b)
    return x

def x_lu(theta,b):
    n = b.shape[0]
    A = A_sparse(theta, n)
    x = torch_sparse_solve.solve(A.unsqueeze(0), b.unsqueeze(1).unsqueeze(0) )
    x = x.squeeze(0).squeeze(-1)
    return x

def loss(x):
    # loss function
    return torch.linalg.norm(x)**2


# Timing and data collection
ns = []
dense_times = []
sparse_times = []
lu_times = []

for n in (2 ** i for i in range(8, 14)):
    torch.manual_seed(0)
    b = torch.rand(n, requires_grad=False, dtype=torch.double)
    theta = torch.rand(n, requires_grad=True, dtype=torch.double)

    for _ in range(2):
        theta.grad = None
        start = time.time()
        f = loss(x_dense(theta, b))
        f.backward()
        dfdtheta_dense = theta.grad.clone().detach()
        t_dense = time.time() - start

    for _ in range(2):
        theta.grad = None
        start = time.time()
        f = loss(x_sparse(theta, b))
        f.backward()
        dfdtheta_sparse = theta.grad.clone().detach()
        t_sparse = time.time() - start

    assert torch.allclose(dfdtheta_dense, dfdtheta_sparse, atol=1e-6), f"Gradient mismatch for n={n}"

    if torch_sparse_solve_available:
        for _ in range(2):
            theta.grad = None
            start = time.time()
            f = loss(x_lu(theta, b))
            f.backward()
            dfdtheta_lu = theta.grad.clone().detach()
            t_lu = time.time() - start

        assert torch.allclose(dfdtheta_dense, dfdtheta_lu, atol=1e-6), f"Gradient mismatch for n={n}"

    ns.append(n)
    dense_times.append(t_dense)
    sparse_times.append(t_sparse)
    if torch_sparse_solve_available:
        lu_times.append(t_lu)


# The rest is all plotting

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.facecolor': '1.0',
    'axes.facecolor': '0.95',
    'grid.color': 'white',
    'grid.linestyle': '-',
    'grid.linewidth': 1.2
})


# Plot
ns = np.array(ns)
dense_times = np.array(dense_times)
sparse_times = np.array(sparse_times)

fig, ax = plt.subplots(figsize=(8, 6))

ax.loglog(ns, dense_times, 'o-', label='torch', linewidth=3)
ax.loglog(ns, sparse_times, 's-', label='SparseSolve', linewidth=3)
if torch_sparse_solve_available:
    ax.loglog(ns, lu_times, '^-', label='torch_sparse_solve', linewidth=3)

# Guide lines and text annotations
x0 = ns[0]
y0 = sparse_times[0]
orders = [1, 2, 3]
labels = [r'$\mathcal{O}(n)$', r'$\mathcal{O}(n^2)$', r'$\mathcal{O}(n^3)$']

for p, label in zip(orders, labels):
    guide_y = y0 * (ns / x0) ** p
    ax.loglog(ns, guide_y, '--', color='black')

    x_last = ns[-1]
    y_last = guide_y[-1]

    # Shift label slightly up and left (log scale aware)
    x_shift = x_last / 1.1
    y_shift = y_last * 0.9

    ax.text(x_shift, y_shift, label,
            color='black', fontsize=14,
            ha='right', va='bottom')

# Styling
ax.set_xlabel('Problem size $n$')
ax.set_ylabel('Time (s)')
ax.set_title(r'∂|A(θ)⁻¹ b|²/∂θ computation time')
ax.legend()
ax.grid(axis='y', which='major')
ax.xaxis.grid(False)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("benchmark.png", dpi=300)
