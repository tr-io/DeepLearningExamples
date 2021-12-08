from hadamard import random_hadamard_encode, random_hadamard_decode

import numpy as np
import pandas as pd
import torch
import math
import time
import matplotlib.pyplot as plt

"""
device = 'cuda' if torch.cuda.is_available() else 'cpu'

seed = 42

sgen = torch.Generator(device=device)
sgen.manual_seed(seed)

rgen = torch.Generator(device=device)
rgen.manual_seed(seed)

time_file_loc = "/home/ubuntu/hadamard-time.txt"
time_X = [10, 100, 10**3, 10**4, 10**5, 10**6, 2*(10**6), 3*(10**6), 4*(10**6), 5*(10**6), 6*(10**6), 7*(10**6), 8*(10**6), 9*(10**6), 10**7, 2*(10**7), 3*(10**7), 4*(10**7), 5*(10**7), 6*(10**7), 7*(10**7), 8*(10**7), 9*(10**7), 10**8]
time_Y = []
time_sterrs = []
# Timing
for i in time_X:
    print(i)
    dim = i
    times = []
    for i in range(100):
        t = torch.rand(dim).to(device)
        vec = t.clone()
        start_time = time.time()
        h_vec = random_hadamard_encode(vec, dim, prng=sgen)
        d_vec = random_hadamard_decode(h_vec, dim, prng=rgen)
        end_time = time.time()
        times.append(end_time - start_time)
    
    times = np.array(times)
    avg_time = np.average(times)
    time_Y.append(avg_time)
    n = times.size
    sterr = np.std(times) / float(math.sqrt(n))
    time_sterrs.append(sterr)
"""

df = pd.read_csv("/home/ubuntu/hadamard-time.csv", index_col=0, header=0)

time_X = range(1, len(df["pow10"].to_numpy()) + 1)
time_Y = df["time"].to_numpy()
time_sterrs = df["sterrs"].to_numpy()

xticks = ['10^1', '10^2', '10^3', '10^4', '10^5', '10^6', '2*(10^6)', '3*(10^6)', '4*(10^6)', '5*(10^6)', '6*(10^6)', '7*(10^6)', '8*(10^6)', '9*(10^6)', '10^7', '2*(10^7)', '3*(10^7)', '4*(10^7)', '5*(10^7)', '6*(10^7)', '7*(10^7)', '8*(10^7)', '9*(10^7)', '10^8']
fig, ax = plt.subplots(figsize=(20, 10))
ax.errorbar(time_X, time_Y, xerr=time_sterrs, fmt='o', ls='none', capsize=10, ecolor='r')
ax.set_title("Hadamard encode/decode time taken")
ax.set_xlabel("Number of elements in vector")
ax.set_xticks(time_X)
ax.set_xticklabels(xticks, rotation='vertical')
ax.set_ylabel("Time taken (seconds)")

fig.savefig("/home/ubuntu/hadamard-times.png")

"""
df = pd.DataFrame()
df['pow10'] = time_X
df['time'] = time_Y
df['sterrs'] = time_sterrs

df.to_csv(time_file_loc)
"""

# Verification stuff
# Verifying vectors are the same after encode + decode
"""
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
"""