from hadamard import random_hadamard_encode, random_hadamard_decode

import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

seed = 42

sgen = torch.Generator(device=device)
sgen.manual_seed(seed)

rgen = torch.Generator(device=device)
rgen.manual_seed(seed)

d = 2**4

t = torch.rand(d).to(device)

vec = t.clone()
print(f"Initial vector: {vec}")
dim = len(vec)

h_vec = random_hadamard_encode(vec, dim, prng=sgen)

d_vec = random_hadamard_decode(h_vec, dim, prng=rgen)
print(f"Restored vector: {d_vec}")

print(torch.eq(d_vec, vec))

assert torch.allclose(d_vec, vec), "Not equal"
