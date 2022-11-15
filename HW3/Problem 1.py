"""
ACMS 80770-03: Deep Learning with Graphs
Instructor: Navid Shervani-Tabar
Fall 2022
University of Notre Dame

Homework 2: Programming assignment
Problem 1
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader

import warnings

import numpy as np
import math
from matplotlib import pyplot as plt

warnings.simplefilter(action='ignore', category=UserWarning)
from chainer_chemistry import datasets
from chainer_chemistry.dataset.preprocessors.ggnn_preprocessor import GGNNPreprocessor

"""
    load data
"""
dataset_raw = datasets.get_qm9(GGNNPreprocessor(kekulize=True),
                               target_index=np.random.choice(range(133000), 6000, False))
# randomly choose 6000 molecules from qm9, dataset_smiles not used here
# 1. atomic number
# 2. adjacency matrix bonding type: single, double, triple, aromatic
# 3. label feature ['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']

V = 9
atom_types = [6, 8, 7, 9, 1]


def adj(x):
    x = x[1]
    adjacency = np.zeros((V, V)).astype(float)
    adjacency[:len(x[0]), :len(x[0])] = x[0] + 2 * x[1] + 3 * x[2]
    return torch.tensor(adjacency)


def sig(x):
    x = x[0]
    atoms = np.ones((V)).astype(float)
    atoms[:len(x)] = x
    out = np.array([int(atom == atom_type) for atom_type in atom_types for atom in atoms]).astype(float)
    return torch.tensor(out).reshape(5, len(atoms)).T


def target(x):
    x = x[2]
    return torch.tensor(x)


adjs = torch.stack(list(map(adj, dataset_raw)))
sigs = torch.stack(list(map(sig, dataset_raw)))
prop = torch.stack(list(map(target, dataset_raw)))[:, 5]
# the 5th label feature is homo feature

train_dataset_adjs = adjs[:4000]
train_dataset_sigs = sigs[:4000]
train_dataset_prop = prop[:4000]

validation_dataset_adjs = adjs[4000:5000]
validation_dataset_sigs = sigs[4000:5000]
validation_dataset_prop = prop[4000:5000]

test_dataset_adjs = adjs[5000:]
test_dataset_sigs = sigs[5000:]
test_dataset_prop = prop[5000:]


class MyDataset(Dataset):
    def __init__(self, adjs, sigs, prop):
        self.adjs = adjs
        self.sigs = sigs
        self.prop = prop

    def __getitem__(self, index):
        adjs = self.adjs[index]
        sigs = self.sigs[index]
        prop = self.prop[index]
        return adjs, sigs, prop

    def __len__(self):
        return len(self.adjs)


train_dataset = MyDataset(train_dataset_adjs, train_dataset_sigs, train_dataset_prop)
validation_dataset = MyDataset(validation_dataset_adjs, validation_dataset_sigs, validation_dataset_prop)
test_dataset = MyDataset(test_dataset_adjs, test_dataset_sigs, test_dataset_prop)


print('Training set has {} instances'.format(len(train_dataset)))
print('Validation set has {} instances'.format(len(validation_dataset)))
print('Test set has {} instances'.format(len(test_dataset)))


train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, drop_last=False, num_workers=0)


def saveModel():
    path = "./NetModel.pth"
    torch.save(model.state_dict(), path)


class GCN(nn.Module):
    """
        Graph convolutional layer
    """

    def __init__(self):
        super(GCN, self).__init__()
        # -- initialize weight
        self.in_features = 5
        self.out_features = 5
        self.weight = Parameter(torch.FloatTensor(5, 5))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, A, H):
        # -- GCN propagation rule
        A_hat_np = A.numpy() + np.identity(n=A.shape[0])  # 9 * 9
        D_hat_np = np.squeeze(np.sum(np.array(A_hat_np), axis=1))
        D_hat_inv_sqrt_np = np.diag(np.power(D_hat_np, -1 / 2))
        A_norm = torch.from_numpy(np.dot(np.dot(D_hat_inv_sqrt_np, A_hat_np), D_hat_inv_sqrt_np))
        HW = torch.mm(H.float(), self.weight)  # weight dtype float
        AHW = torch.mm(A_norm.float(), HW)
        # -- non-linearity
        return F.relu(AHW)


class GraphPooling(nn.Module):
    """
        Graph pooling layer
    """

    def __init__(self):
        super(GraphPooling, self).__init__()

    def forward(self, H):
        # -- multi-set pooling operator
        x = H.sum(dim=0)
        return x


class MyModel(nn.Module):
    """
        Regression  model
    """

    def __init__(self):
        super(MyModel, self).__init__()
        # -- initialize layers
        self.gc = GCN()
        self.pooling = GraphPooling()
        self.fc = torch.nn.Linear(5, 1)

    def forward(self, A, h0):
        x = self.gc(A, h0)
        x = self.pooling(x)
        x = self.fc(x)
        return x


"""
    Train
"""
# -- Initialize the model, loss function, and the optimizer
model = MyModel()
MyLoss = nn.MSELoss()
MyOptimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# -- update parameters
losses = []
for epoch in range(200):
    print("Epoch:", epoch)
    running_loss = 0.0
    model.train()
    for i, data in enumerate(train_loader):
        adj, sig, prop = data

        MyOptimizer.zero_grad()
        # -- predict
        pred = model(adj[i], sig[i])

        # -- loss
        loss = MyLoss(pred, prop[i])
        loss.backward()

        # -- optimize
        MyOptimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / 100
    losses.append(epoch_loss)

train_loss_value = sum(losses) / 200

# -- plot loss

plt.plot(losses)
plt.show()

# -- validation
with torch.no_grad():
    model.eval()
    running_vall_loss = 0
    for i in range(len(validation_dataset)):
        adj, sig, prop = validation_dataset[i][0], validation_dataset[i][1], validation_dataset[i][2]
        prediction = model(adj, sig)
        target = prop
        running_vall_loss += MyLoss(prediction, target).item()
    val_loss_value = running_vall_loss / len(validation_dataset)

print('Completed training batch', 200,
      'Training Loss is: %.4f' % train_loss_value,
      'Validation Loss is: %.4f' % val_loss_value)

saveModel()

"""
    Test
"""


with torch.no_grad():
    model.eval()
    pred = []
    targ = []
    for i in range(len(test_dataset)):
        adj, sig, prop = test_dataset[i][0], test_dataset[i][1], test_dataset[i][2]
        prediction = model(adj, sig)
        pred.append(prediction.detach().numpy())
        target = prop
        targ.append(prop.detach().numpy())

plt.scatter(pred, targ)
plt.show()