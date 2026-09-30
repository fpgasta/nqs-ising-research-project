# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "netket>=3.22.4",
# ]
# ///

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import netket as nk
from netket.operator.spin import sigmax, sigmaz
import jax
from scipy.sparse.linalg import eigsh
import jax.numpy as jnp
from flax import nnx

N = 20
hi = nk.hilbert.Spin(s=1/2, N=N)

# print(hi.random_state(jax.random.key(0), 3))

Gamma = -1
H = sum([Gamma * sigmax(hi, i) for i in range(N)])
V = -1
H += sum([V * sigmaz(hi, i) @ sigmaz(hi, (i+1) % N) for i in range(N)])

sp_h = H.to_sparse()
eig_vals, eig_vecs = eigsh(sp_h, k=2, which="SA")
print("eigenvalues with scipy sparse:",eig_vals)
E_gs = eig_vals[0]
