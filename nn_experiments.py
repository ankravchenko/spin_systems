from spin_lattices import KagomeLattice, SpinLattice, SquareLattice, TriangleLattice
from heisenberg_hamiltonians import HeisenbergJ1J2, SpinSystem
from boolean_analysis import (
    BooleanFourierAnalyzer,
    keep_largest_n,
    keep_everything,
    ScorerType,
    get_scorer,
    SignalOption,
    AmplitudeMedianBinSignalKind,
    SignSignalKind,
    AmplitudeSignalKind,
    SignalKind,
)
from boolean_fourier_learner import BooleanFourierLearner

LATTICE_TYPE='square'
EPOCHS_N = 200

from pathlib import Path
import numpy as np
import pandas as pd
import lattice_symmetries as ls
import matplotlib.pyplot as plt
from heisenberg_hamiltonians import batched_state_info_df
from itertools import product
#import numpy.typing as npt
from tqdm import tqdm
import seaborn as sns
import parse

from parity import popcount

import os
from os import path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

import pickle


import matplotlib.pyplot as plt
import seaborn as sns
import random as rd
from sklearn.preprocessing import StandardScaler

import itertools


import umap.umap_ as umap
reducer = umap.UMAP()

#насколько свободен стандартный индекс относительно случайных преобразований
#из какого класса преобразование, в какую размерность это переходит (пропорции матрица)
#UMAP of the initial sets. are they 
#look at first several epochs (UMAP representations)
#does simple XOR become more separable with random transformations
#DSI for representations (separability index)

ground_state_cache_dir = Path("groundstates")
fourier_learners_cache_dir = Path("fourier_learners_cache")
experiments_dir = Path("experiments") / "kagome-24-nn-2023-01-26"
experiments_dir.mkdir(parents=True, exist_ok=True)

def get_inputs_and_labels(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X = torch.tensor(df[features].values.astype("float"), dtype=torch.float32)
    y = torch.tensor(df["y"].values.astype("int8"), dtype=torch.long)
    probs = torch.tensor(df["prob"].values.astype("float"), dtype=torch.float32)
    return X, y, probs


def evaluate(net, inputs, labels, probs):
    with torch.no_grad():
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / len(labels)

        sign_overlap = (
            ((predicted * 2 - 1) * (labels * 2 - 1) * probs).sum() / probs.sum()
        ).item()
        return accuracy, sign_overlap

#square
J2 = 0
system = HeisenbergJ1J2(
    lattice=SquareLattice(width=6, height=4),
    J1=1,
    J2=J2,
    use_symmetries=True,
    spin_inversion=1,
    ground_state_cache_dir=ground_state_cache_dir,
)

'''
#kagome
J2 = 1
system = HeisenbergJ1J2(
    lattice=KagomeLattice(width=2, height=4),
    J1=1,
    J2=J2,
    use_symmetries=True,
    spin_inversion=1,
    ground_state_cache_dir=ground_state_cache_dir,
)
'''
'''
#triangle
J2 = 1
system = HeisenbergJ1J2(
    lattice=TriangleLattice(width=2, height=4),
    J1=1,
    J2=J2,
    use_symmetries=True,
    spin_inversion=1,
    ground_state_cache_dir=ground_state_cache_dir,
)
'''
system.get_eigenstates(1)
'''analyzer = BooleanFourierAnalyzer(
    system=system,
    use_subset_symmetries=True,
    show_progress=True,
    cache_dir=fourier_learners_cache_dir,
)
'''
df = (
    system.get_df_ground_state(
        canonical_basis=True, unpack_configurations=True, expand_basis_columns=True
    )
    .assign(
        sign=(lambda df: np.sign(df["eigenstate_coeff"])),
        prob=(lambda df: np.abs(df["eigenstate_coeff"]) ** 2),
    )
    .assign(y=lambda df: (df["sign"] == 1).astype(int))
)


eps_train = 1e-2
val_eps = 1e-2
test_eps = 1e-2
batch_size = 64

df_train = df.sample(frac=eps_train, weights="prob")
df_val = df.drop(df_train.index).sample(frac=val_eps, weights="prob")
df_test = df.drop(df_train.index).drop(df_val.index).sample(frac=test_eps, weights="prob")

n_batches = int(np.ceil(len(df_train) / batch_size))
epochs = EPOCHS_N #20000
#################
node_out = {}
node_in = {}

#function to generate hook function for each module
def get_node_out(name):
  def hook(model, input, output):
    node_in[name] = input[0][0].detach()
    node_out[name] = output[0].detach()
  return hook
###################
net = nn.Sequential(
    nn.Linear(system.number_spins, 64),
    nn.ReLU(),
    nn.Linear(64, 2),
)

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

hook_handles = {}
for name, module in net.named_modules():
    hook_handles[name] = module.register_forward_hook(get_activation(name))


'''

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.l1 = nn.Linear(system.number_spins, 64),
        #self.relu = nn.ReLU()
        self.l2 = nn.Linear(64, 2)

    
    def forward(self, x):
        x = F.relu(self.l1(x))
        #x = F.relu(self.relu(x))
        x = F.relu(self.l2(x))
        return x

net=net()
'''
'''
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
'''

if path.exists(LATTICE_TYPE+"_FI.log"):
    os.remove(LATTICE_TYPE+"_FI.log")
logfile_fi  = open(LATTICE_TYPE+"_FI.log", "a", newline='')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
features = [f"s{i}" for i in range(system.number_spins)]


inputs_val, labels_val, probs_val = get_inputs_and_labels(df_val)
'''
w0 = list(net.parameters())
w0_list=w0[2].detach().numpy() #[x.detach().numpy() for x in w0]
'''
w0=[net[0].weight.detach().clone().numpy(), net[2].weight.detach().clone().numpy()]

for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    i = None
    loss = None

    for i in range(n_batches):
        data = df_train.iloc[i * batch_size : (i + 1) * batch_size]
        inputs, labels, probs = get_inputs_and_labels(data)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        '''
        w = list(net.parameters())
        w_list=w[2].detach().numpy() #[x.detach().numpy() for x in w]
        '''
    w=[net[0].weight.detach().clone().numpy(), net[2].weight.detach().clone().numpy()]
    diff = [a - b for a, b in zip(w, w0)]
    grads_reshape = [v.flatten() for v in diff]
    grads_pow2 = [pow(v,2) for v in grads_reshape]
    grads_sum=[sum(a) for a in grads_pow2]
    grads_fsum=sum(grads_sum)
    #print('FI: ', str(grads_fsum))
    w0=w
    #w0_list=[x.detach().numpy() for x in w0]
    logfile_fi.write(str(grads_fsum))
    logfile_fi.write('	')
    print('FI: ', grads_fsum)
    logfile_fi.flush()

    '''
    a=activation['0']
    print(a)
    '''
    print(f"[{epoch + 1}, {i}] loss: {loss}")
    
    accuracy, sign_overlap = evaluate(net, inputs, labels, probs)
    print(f"Train set: accuracy: {100 * accuracy} %, sign overlap: {sign_overlap}")

    accuracy_val, sign_overlap_val = evaluate(net, inputs_val, labels_val, probs_val)
    print(f"Validation set: accuracy: {100 * accuracy_val} %, sign overlap: {sign_overlap_val}")

    '''
    a=activation['0']
    print(a)
    print('##############')'''

logfile_fi.close()
layer0=activation['0'].numpy()
layer1=activation['1'].numpy()
layer2=activation['2'].numpy()



