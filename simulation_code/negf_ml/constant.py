import math
import torch

lattice_n = 32
lattice_w = 24
kT = 0.0025
dmp = 0.55
dt = 0.5
total_round = 5000
init = "data_input/c0.dat"
cut = 5
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

