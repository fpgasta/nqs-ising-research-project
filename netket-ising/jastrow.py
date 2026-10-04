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



class JasShort(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.j1 = nnx.Param(0.01 * jax.random.normal(rngs.params(), (1,)), dtype=float)
        self.j2 = nnx.Param(0.01 * jax.random.normal(rngs.params(), (1,)), dtype=float)

    def __call__(self, x: jax.Array):
        corr1 = x * jnp.roll(x, -1, axis=-1)
        corr2 = x * jnp.roll(x, -2, axis=-1)
        return jnp.sum(self.j1 * corr1 + self.j2 * corr2, axis=-1)



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
model = JasShort(rngs=nnx.Rngs(1))
sampler = nk.sampler.MetropolisLocal(hi)
vstate = nk.vqs.MCState(sampler, model, n_samples=1008)
# print(vstate.parameters)
# E = vstate.expect(H)
# # print(E)
# print("Mean                  :", E.mean)
# print("Error                 :", E.error_of_mean)
# print("Variance              :", E.variance)
# print("Convergence indicator :", E.R_hat)
# print("Correlation time      :", E.tau_corr)


# optimization with preconditioner and logging
optimizer = nk.optimizer.Sgd(learning_rate=0.01)

gs = nk.driver.VMC(H, optimizer, variational_state=vstate, preconditioner=nk.optimizer.SR(diag_shift=0.1))
log = nk.logging.RuntimeLog()
gs.run(n_iter=300, out=log)

jas_energy = vstate.expect(H)
error = abs((jas_energy.mean - eig_vals[0]) / eig_vals[0])
print(f"Optimized energy : {jas_energy}")
print(f"relative error   : {error}")

data_jastrow = log.data
print(data_jastrow)

plt.errorbar(
    data_jastrow["Energy"].iters,
    data_jastrow["Energy"].Mean,
    yerr=data_jastrow["Energy"].Sigma
)
plt.xlabel("Iterations")
plt.ylabel("Energy")
plt.show()