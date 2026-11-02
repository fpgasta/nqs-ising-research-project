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


class RBM(nnx.Module):
    def __init__(self, N: int, *, rngs: nnx.Rngs):
        key = rngs.params()
        # self.a = nnx.Param(jax.random.normal(rngs.params(), (N,)), dtype=float)
        self.a = nnx.Param(jnp.zeros((N,)))
        nh = N
        # self.b = nnx.Param(jax.random.normal(rngs.params(), (nh,)), dtype=float)
        self.b = nnx.Param(jnp.zeros((N,)))
        self.W = nnx.Param(1/sqrt(N) * jax.random.normal(rngs.params(), (nh,N)), dtype=float)

    def __call__(self, x: jax.Array):
        log_amplitude = jnp.sum(self.a * x, axis=-1)
        # print(jnp.cosh( x @ self.W.T + self.b ).shape)
        # print(self.a.shape)
        log_amplitude += jnp.sum(jnp.log(2 * jnp.cosh( x @ self.W.T + self.b )), axis=-1) # does this work in terms of dimensions?
        return log_amplitude






N = 20
hi = nk.hilbert.Spin(s=1/2, N=N)
theta = 1/4 * jnp.pi
Gamma = jnp.cos(theta)
# Gamma = -1
H = sum([Gamma * sigmaz(hi, i) for i in range(N)])
V = jnp.sin(theta)
# V = -1
H += sum([V * sigmax(hi, i) @ sigmax(hi, (i+1) % N) for i in range(N)])

# # exact solution
sp_h = H.to_sparse()
eig_vals, eig_vecs = eigsh(sp_h, k=2, which="SA")
E_gs = eig_vals[0]

# mean field ansatz
model = RBM(N=N, rngs=nnx.Rngs(2))


sampler = nk.sampler.MetropolisLocal(hi)
vstate = nk.vqs.MCState(sampler, model, n_samples=1008)
print("n params:", vstate.n_parameters)


# optimization with preconditioner and logging
optimizer = nk.optimizer.Sgd(learning_rate=0.05)

gs = nk.driver.VMC(H, optimizer, variational_state=vstate, preconditioner=nk.optimizer.SR(diag_shift=0.1))
log = nk.logging.RuntimeLog()
gs.run(n_iter=300, out=log)

rbm_energy = vstate.expect(H)
# error = abs((rbm_energy.mean - eig_vals[0]) / eig_vals[0])
print(f"Optimized energy : {rbm_energy}")
# print(f"relative error   : {error}")

data_RBM = log.data

plt.errorbar(
    data_RBM["Energy"].iters,
    data_RBM["Energy"].Mean,
    yerr=data_RBM["Energy"].Sigma,
    label="RBM"
)
plt.hlines([E_gs], xmin=0, xmax=300, color="black", label="Exact=%.2f"%E_gs)
plt.legend()

plt.xlabel("Iterations")
plt.ylabel("Energy")
plt.show()