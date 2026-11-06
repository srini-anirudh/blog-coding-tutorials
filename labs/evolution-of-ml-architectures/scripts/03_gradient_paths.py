"""Every family's failure is a product of Jacobians. Measure the product.

1. Deep stacks: sigmoid, tanh and ReLU, plain and residual (y = x + F(x)).
2. Elman RNN: ||dh_T/dh_{T-k}|| for sigma_max(W_h) = 0.9, 1.0, 1.1, linear and tanh.
3. LSTM: the cell path prod diag(f_t), for forget-gate bias 0, +1 and +3, and the full Jacobian.
4. Pre-LN against post-LN: weight-gradient norm per block at initialisation.
"""

import _setup  # noqa: F401

from archlab.env import start
from archlab.gradients import deep_stack_input_grad, lstm_cell_path, lstm_full_jacobian, norm_placement_grads
from archlab.gradients import rnn_jacobian_norms

start("03_gradient_paths")

print("1. ||dL/dx|| / ||dL/dy|| through a stack of Linear + activation, width 64, PyTorch default init")
print(f"   {'activation':>10} {'residual':>8} " + " ".join(f"{f'depth {n}':>10}" for n in (6, 20, 50)))
for act in ["sigmoid", "tanh", "relu"]:
    for res in [False, True]:
        vals = [deep_stack_input_grad(n, 64, act, res) for n in (6, 20, 50)]
        print(f"   {act:>10} {str(res):>8} " + " ".join(f"{v:>10.2e}" for v in vals))
print(f"   The activation alone: sigma'(z) <= 1/4, so six sigmoids multiply by at most {0.25 ** 6:.2e}")
print()

ks = [1, 10, 50, 100]
print("2. Elman RNN, d = 64, W_h = lambda x orthogonal: ||dh_T / dh_{T-k}||_2")
print(f"   {'lambda':>6} {'tanh':>5} " + " ".join(f"{f'k={k}':>10}" for k in ks) + f" {'lambda^100':>11}")
for lam in [0.9, 1.0, 1.1]:
    for nonlin in [False, True]:
        n = rnn_jacobian_norms(lam, 100, nonlinear=nonlin)
        print(f"   {lam:>6} {str(nonlin):>5} " + " ".join(f"{n[k - 1].item():>10.2e}" for k in ks)
              + f" {lam ** 100:>11.2e}")
print("   Linear rows equal lambda^k exactly. tanh' <= 1 only shrinks it, so lambda = 1.1 still vanishes with tanh here.")
print()

print("3. LSTM, d = 64, random weights")
print(f"   {'forget bias':>11} " + " ".join(f"{f'k={k}':>10}" for k in ks) + "   (cell path: max of prod f)")
for fb in [0.0, 1.0, 3.0]:
    p = lstm_cell_path(fb, 100)
    print(f"   {fb:>11} " + " ".join(f"{p[k - 1].item():>10.2e}" for k in ks))
print(f"   {'forget bias':>11} " + " ".join(f"{f'k={k}':>10}" for k in ks) + "   (full Jacobian dc_T/dc_{T-k}, autograd)")
for fb in [0.0, 1.0, 3.0]:
    full = lstm_full_jacobian(fb, ks)
    print(f"   {fb:>11} " + " ".join(f"{v:>10.2e}" for v in full))
print()

print("4. Weight-gradient norm per block at init, 24 blocks, d = 64 (block 1 = bottom)")
for pre in [True, False]:
    g = norm_placement_grads(24, pre)
    label = "pre-LN " if pre else "post-LN"
    print(f"   {label}: block 1 {g[0]:.2e}  block 12 {g[11]:.2e}  block 24 {g[-1]:.2e}  max/min {max(g) / min(g):.1f}")
print("   At init this toy shows no depth effect for either placement. The post-LN warmup problem appears during")
print("   training with a large learning rate, which this script does not run.")
