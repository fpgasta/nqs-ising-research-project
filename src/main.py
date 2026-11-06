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


class myRBM(nnx.Module):
    def __init__(self, N: int, alpha: int=1, *, rngs: nnx.Rngs):
        key = rngs.params()
        self.a = nnx.Param(0.01 * jax.random.normal(rngs.params(), (N,)), dtype=float)
        nh = alpha * N
        self.b = nnx.Param(0.01 * jax.random.normal(rngs.params(), (nh,)), dtype=float)
        self.W = nnx.Param(0.01 * jax.random.normal(rngs.params(), (nh,N)), dtype=float)

    def __call__(self, x: jax.Array):
        log_amplitude = jnp.sum(self.a * x, axis=-1)
        # print(jnp.cosh( x @ self.W.T + self.b ).shape)
        # print(self.a.shape)
        log_amplitude += jnp.sum(jnp.log(2 * jnp.cosh( x @ self.W.T + self.b )), axis=-1) # does this work in terms of dimensions?
        return log_amplitude

class JasShort(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.j1 = nnx.Param(0.01 * jax.random.normal(rngs.params(), (1,)), dtype=float)
        self.j2 = nnx.Param(0.01 * jax.random.normal(rngs.params(), (1,)), dtype=float)

    def __call__(self, x: jax.Array):
        corr1 = x * jnp.roll(x, -1, axis=-1)
        corr2 = x * jnp.roll(x, -2, axis=-1)
        return jnp.sum(self.j1 * corr1 + self.j2 * corr2, axis=-1)


class NQS_ising():
    def __init__(self, N, h, J, model, interaction_direction: string='z', exact_solution=False):
        self.N = N
        self.exact_solution = exact_solution
        hi = nk.hilbert.Spin(s=1/2, N=N)

        if interaction_direction == 'x':
            self.H = sum([h * sigmaz(hi, i) for i in range(N)])
            self.H += sum([J * sigmax(hi, i) @ sigmax(hi, (i+1) % N) for i in range(N)])
        elif interaction_direction == 'z':
            self.H = sum([h * sigmax(hi, i) for i in range(N)])
            self.H += sum([J * sigmaz(hi, i) @ sigmaz(hi, (i+1) % N) for i in range(N)])
        else:
            print("ERROR: invalid interaction direction", interaction_direction)
            return

        self.sampler = nk.sampler.MetropolisLocal(hi)
        self.vstate = nk.vqs.MCState(self.sampler, model, n_samples=1008)
        # self.vstate = nk.vqs.MCState(self.sampler, model, n_samples=4032)
        print("n params:", self.vstate.n_parameters)

        self.optimized_energy = None

    def solve_exactly(self):
        sp_h = self.H.to_sparse()
        eig_vals = nk.exact.lanczos_ed(self.H, k=1)
        self.E_gs = eig_vals[0]

    def optimization_loop(self, n_iter: int=300, learning_rate=0.05):
        self.n_iter = n_iter
        if self.exact_solution:
            self.solve_exactly()

        optimizer = nk.optimizer.Sgd(learning_rate=learning_rate)

        gs = nk.driver.VMC(self.H, optimizer, variational_state=self.vstate, preconditioner=nk.optimizer.SR(diag_shift=0.1))
        log = nk.logging.RuntimeLog()
        gs.run(n_iter=n_iter, out=log)

        self.optimized_energy = self.vstate.expect(self.H)
        self.optimization_data = log.data

    def plot(self):
        if self.optimized_energy is None:
            print("ERROR: optimization loop hasn't been run.")
            return

        print(f"Optimized energy : {self.optimized_energy}")
        if self.exact_solution:
            error = abs((self.optimized_energy.mean - self.E_gs) / self.E_gs)
            print(f"relative error   : {error}")

        plt.errorbar(
            self.optimization_data["Energy"].iters,
            self.optimization_data["Energy"].Mean,
            yerr=self.optimization_data["Energy"].Sigma,
            label="NQS"
        )
        if self.exact_solution:
            plt.hlines([self.E_gs], xmin=0, xmax=self.n_iter, color="black", label="Exact=%.2f"%self.E_gs)
        plt.legend()

        plt.xlabel("Iterations")
        plt.ylabel("Energy")
        # plt.ylim(-17,-4)
        # plt.show()
        plt.savefig("plot/energy.pdf", bbox_inches="tight")



N = 16



### plot infidelity of ground states for different parameters
# irange = range(-3, 4)
# ground_states = []
# for i in tqdm(irange):
#     theta = (5/4 + i/16) * jnp.pi
#     print("i=%i, theta=%.2f" % (i, theta))
#     h = jnp.cos(theta)
#     J = jnp.sin(theta)

#     model = RBM(N=N, rngs=nnx.Rngs(2))
#     nqs_ising = NQS_ising(N, h, J, model, 'z', False)
#     nqs_ising.optimization_loop(n_iter=300)

#     if i == 0:
#         infidelity_operator = nk.observable.InfidelityOperator(nqs_ising.vstate)

#     ground_states.append(nqs_ising.vstate)


# infidelities = []
# for i in irange:
#     infidelities.append(ground_states[i+3].expect(infidelity_operator).mean)

# plt.plot(irange, infidelities)
# plt.show()



### calculate only one ground state
theta = (5/4) * jnp.pi
# h = jnp.cos(theta)
# J = jnp.sin(theta)
J = +0.3
h = -1

# model = myRBM(N=N, alpha=1, rngs=nnx.Rngs(2))
# model = nk.models.RBM(alpha=1)
model = JasShort(rngs=nnx.Rngs(2))
nqs_ising = NQS_ising(N, h, J, model, 'z', True)
nqs_ising.optimization_loop(n_iter=500, learning_rate=0.01)
nqs_ising.plot()