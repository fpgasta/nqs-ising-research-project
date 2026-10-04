import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import netket as nk
from netket.operator.spin import sigmax, sigmaz
from scipy.sparse.linalg import eigsh
import jax
import jax.numpy as jnp
from flax import nnx
import flax.linen as nn
import netket.nn as nknn
from tqdm import tqdm
import matplotlib.pyplot as plt



class SymmModel(nnx.Module):
    def __init__(self, N: int, alpha: int=1, *, rngs: nnx.Rngs):
        self.alpha = alpha
        # linear (dense) neural network layer
        dense_symm_linen = nknn.DenseSymm(
            symmetries=graph.translation_group(),
            features=alpha,
            kernel_init=nn.initializers.normal(stddev=0.01),
        )
        self.linear_symm = nnx.bridge.ToNNX(dense_symm_linen, rngs=rngs).lazy_init(
            jnp.ones((1, 1, N))
        )

    def __call__(self, x: jax.Array):
        x = x.reshape(-1, 1, x.shape[-1])

        x = self.linear_symm(x)
        x = nnx.relu(x)
        return jnp.sum(x, axis=(-1, -2))



N = 20
hi = nk.hilbert.Spin(s=1/2, N=N)

# define graph
graph = nk.graph.Chain(length=N, pbc=True)
print(graph.translation_group())

# define hamiltonian using graph
Gamma = -1
H = sum([Gamma * sigmax(hi, i) for i in range(N)])
V = -1
H += sum([V * sigmaz(hi, i) @ sigmaz(hi, j) for (i, j) in graph.edges()])

# exact solution
sp_h = H.to_sparse()
eig_vals, eig_vecs = eigsh(sp_h, k=2, which="SA")
E_gs = eig_vals[0]



# symmetry model
# Let us define a model with 4 features per symmetry
model = SymmModel(N=N, alpha=4, rngs=nnx.Rngs(2))
sampler = nk.sampler.MetropolisLocal(hi)
vstate = nk.vqs.MCState(sampler, model, n_samples=1008)
print("n params:", vstate.n_parameters)


# optimization with preconditioner and logging
optimizer = nk.optimizer.Sgd(learning_rate=0.1)

gs = nk.driver.VMC(H, optimizer, variational_state=vstate, preconditioner=nk.optimizer.SR(diag_shift=0.1))
log = nk.logging.RuntimeLog()
gs.run(n_iter=600, out=log)

symm_energy = vstate.expect(H)
error = abs((symm_energy.mean - eig_vals[0]) / eig_vals[0])
print(f"Optimized energy : {symm_energy}")
print(f"relative error   : {error}")

data_symm = log.data

plt.errorbar(
    data_symm["Energy"].iters[50:],
    data_symm["Energy"].Mean[50:],
    yerr=data_symm["Energy"].Sigma[50:],
    label="SymmModel"
)
plt.hlines([E_gs], xmin=0, xmax=600, color="black", label="Exact")
plt.legend(frameon=False)

plt.xlabel("Iterations")
plt.ylabel("Energy")
plt.show()