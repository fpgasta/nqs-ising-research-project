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



class MF(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        key = rngs.params()
        self.log_phi_local = nnx.Param(jax.random.normal(key, (1, )))

    def __call__(self, x: jax.Array):
        p = nnx.log_sigmoid(self.log_phi_local * x)
        return 0.5 * jnp.sum(p, axis=-1)



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
mf_model = MF(rngs=nnx.Rngs(0))
sampler = nk.sampler.MetropolisLocal(hi)
vstate = nk.vqs.MCState(sampler, mf_model, n_samples=512)
print(vstate.parameters)
E = vstate.expect(H)
# print(E)
print("Mean                  :", E.mean)
print("Error                 :", E.error_of_mean)
print("Variance              :", E.variance)
print("Convergence indicator :", E.R_hat)
print("Correlation time      :", E.tau_corr)

# # manual optimisation
# energy_history = []
# n_steps = 100

# for i in tqdm(range(n_steps)):
#     E, E_grad = vstate.expect_and_grad(H)
#     energy_history.append(E.mean.real)
#     new_pars = jax.tree_util.tree_map(
#         lambda x, y: x-0.05*y, vstate.parameters, E_grad
#     )
#     vstate.parameters = new_pars

# plt.plot(energy_history)
# plt.show()

# netket optimisation driver
vstate.init_parameters()
optimizer = nk.optimizer.Sgd(learning_rate=0.05)
gs = nk.driver.VMC(H, optimizer, variational_state=vstate)
gs.run(n_iter=300)

mf_energy = vstate.expect(H)
error = abs((mf_energy.mean - eig_vals[0]) / eig_vals[0])
print("Optimized energy and relative error: ", mf_energy, error)