import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import netket as nk
from netket.operator.spin import sigmax, sigmaz
from scipy.sparse.linalg import eigsh
import jax
import jax.numpy as jnp
from flax import nnx
from tqdm import tqdm
import matplotlib.pyplot as plt



class FFN(nnx.Module):
    def __init__(self, N: int, alpha: int=1, *, rngs: nnx.Rngs):
        self.alpha = alpha
        # linear (dense) neural network layer
        self.linear = nnx.Linear(in_features=N, out_features=alpha*N, rngs=rngs)

    def __call__(self, x: jax.Array):
        y = self.linear(x)
        y = nnx.relu(y)
        return jnp.sum(y, axis=-1)



N = 20
hi = nk.hilbert.Spin(s=1/2, N=N)
Gamma = -1
H = sum([Gamma * sigmax(hi, i) for i in range(N)])
V = -1
H += sum([V * sigmaz(hi, i) @ sigmaz(hi, (i+1) % N) for i in range(N)])

# exact solution
sp_h = H.to_sparse()
eig_vals, eig_vecs = eigsh(sp_h, k=2, which="SA")
E_gs = eig_vals[0]

# mean field ansatz
model = FFN(N=N, alpha=1, rngs=nnx.Rngs(2))
sampler = nk.sampler.MetropolisLocal(hi)
vstate = nk.vqs.MCState(sampler, model, n_samples=1008)
print("n params:", vstate.n_parameters)
# print(vstate.parameters)
# E = vstate.expect(H)
# # print(E)
# print("Mean                  :", E.mean)
# print("Error                 :", E.error_of_mean)
# print("Variance              :", E.variance)
# print("Convergence indicator :", E.R_hat)
# print("Correlation time      :", E.tau_corr)


# optimization with preconditioner and logging
optimizer = nk.optimizer.Sgd(learning_rate=0.1)

gs = nk.driver.VMC(H, optimizer, variational_state=vstate, preconditioner=nk.optimizer.SR(diag_shift=0.1))
log = nk.logging.RuntimeLog()
gs.run(n_iter=300, out=log)

ffn_energy = vstate.expect(H)
error = abs((ffn_energy.mean - eig_vals[0]) / eig_vals[0])
print(f"Optimized energy : {ffn_energy}")
print(f"relative error   : {error}")

data_FFN = log.data

plt.errorbar(
    data_FFN["Energy"].iters,
    data_FFN["Energy"].Mean,
    yerr=data_FFN["Energy"].Sigma,
    label="FFN"
)
plt.hlines([E_gs], xmin=0, xmax=300, color="black", label="Exact")
plt.legend()

plt.xlabel("Iterations")
plt.ylabel("Energy")
plt.show()