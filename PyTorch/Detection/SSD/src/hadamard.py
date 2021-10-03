import torch
import numpy as np

##############################################################################
##############################################################################

def pad(vec):    
    
  '''
  Pad dim to a power of 2
  '''
    
  dim = vec.numel()    
  if not dim & (dim-1) == 0:
      dim = int(2**(np.ceil(np.log2(dim))))     
  pvec = torch.zeros(dim).to(vec.device)
  pvec[:vec.numel()] = vec

  return pvec

##############################################################################
##############################################################################

def hadamard_rotate(vec):
    
  '''
  In-place 1D hadamard transform 
  '''
    
  numel = vec.numel()
  if numel & (numel-1) != 0:
      raise Exception("vec numel must be a power of 2")
      
  h = 2

  while h <= numel:
      
    hf = h // 2
    vec = vec.view(numel // h, h)

    vec[:, :hf] = vec[:, :hf] + vec[:, hf:2 * hf]
    vec[:, hf:2 * hf] = vec[:, :hf] - 2 * vec[:, hf:2 * hf]
    h *= 2

  vec /= np.sqrt(numel)

##############################################################################
##############################################################################

def random_hadamard_encode(vec, numel, prng=None):

  ### pad if nedded
  vec = pad(vec)
  
  ### in-place hadamard transform
  if prng is not None:
    vec = vec * (2 * torch.bernoulli(torch.ones(vec.numel(), device=vec.device) / 2, generator=prng) - 1)
  hadamard_rotate(vec)

  #### send
  return vec

##############################################################################

def random_hadamard_decode(vec, numel, prng=None, frac=1):

  ### in-place hadamard transform (inverse)
  hadamard_rotate(vec)
  if prng is not None:
    vec = vec * (2 * torch.bernoulli(torch.ones(vec.numel(), device=vec.device) / 2, generator=prng) - 1)

  ##### return
  return (vec/frac)[:numel]

##############################################################################
##############################################################################
