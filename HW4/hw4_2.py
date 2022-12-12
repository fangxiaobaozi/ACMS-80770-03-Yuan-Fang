"""
ACMS 80770-03: Deep Learning with Graphs
Instructor: Navid Shervani-Tabar
Fall 2022
University of Notre Dame
Homework 4: Programming assignment
Problem 2
"""
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=UserWarning)

from torch import nn
from rdkit import Chem
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
from chainer_chemistry import datasets
from chainer_chemistry.dataset.preprocessors.ggnn_preprocessor import GGNNPreprocessor


# -- load data
class MolecularDataset(Dataset):
    def __init__(self, N, train=True):
        if train:
            start, end = 0, 100000
        else:
            start, end = 100000, 130000

        dataset, dataset_smiles = datasets.get_qm9(GGNNPreprocessor(kekulize=True),
                                                   return_smiles=True,
                                                   target_index=np.random.choice(range(133000)[start:end], N, False))

        self.atom_types = [6, 8, 7, 9, 1]
        self.V = 9

        self.adjs = torch.stack(list(map(self.adj, dataset)))
        self.sigs = torch.stack(list(map(self.sig, dataset)))
        self.prop = torch.stack(list(map(self.target, dataset)))[:, 5]
        self.prop_2 = torch.stack(list(map(self.target_2, dataset_smiles)))

    def target_2(self, smiles):
        """
            compute the number of hydrogen-bond acceptor atoms
        :param smiles: smiles molecular representation
        :return:
        """
        mol = Chem.MolFromSmiles(smiles)

        return torch.tensor(Chem.rdMolDescriptors.CalcNumHBA(mol))

    def adj(self, x):
        x = x[1]
        adjacency = np.zeros((self.V, self.V)).astype(float)
        adjacency[:len(x[0]), :len(x[0])] = x[0] + 2 * x[1] + 3 * x[2]
        return torch.tensor(adjacency)

    def sig(self, x):
        x = x[0]
        atoms = np.ones((self.V)).astype(float)
        atoms[:len(x)] = x
        out = np.array([int(atom == atom_type) for atom_type in self.atom_types for atom in atoms]).astype(float)
        return torch.tensor(out).reshape(5, len(atoms)).T

    def target(self, x):
        """
            return Highest Occupied Molecular Orbital (HOMO) energy
        :param x:
        :return:
        """
        x = x[2]
        return torch.tensor(x)

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, item):
        return self.adjs[item], self.sigs[item], self.prop[item], self.prop_2[item]


class kernel:
    def __init__(self, K, R, d, J, lamb_max):
        # -- filter properties
        self.R = float(R)
        self.J = J
        self.K = K
        self.d = d
        self.lamb_max = torch.tensor(lamb_max)

        # -- Half-Cosine kernel
        self.a = R * np.log(lamb_max) / (J - R + 1)
        self.g_hat = lambda lamb: d[0] + d[1] * np.cos(
            2 * np.pi * (lamb.item() / self.a + 0.5)) if 0 <= -lamb.item() < self.a else 0

    def wavelet(self, lamb, j):
        """
            constructs wavelets ($j\in [2, J]$).
        :param lamb: eigenvalue (analogue of frequency).
        :param j: filter index in the filter bank.
        :return: filter response to input eigenvalues.
        """
        res = []
        for _ in lamb:
            res.append(self.g_hat(np.log(_) - self.a * (j - 1) / self.R))
        return torch.DoubleTensor(res)

    def scaling(self, lamb):
        """
            constructs scaling function (j=1).
        :param lamb: eigenvalue (analogue of frequency).
        :return: filter response to input eigenvalues.
        """
        res = []
        b = 0
        for k in range(1, self.K + 1):
            b += self.d[k] ** 2
        b = self.R / 2 * b
        for _ in lamb:
            c = 0
            for j in range(2, self.J + 1):
                c += abs((self.g_hat(np.log(_) - self.a * (j - 1) / self.R) ** 2))
            e = self.R * self.d[0] ** 2 + b - c
            if abs(e) < 1e-8:
                e = 0
            res.append(e)
        return torch.DoubleTensor(res)


def pooling(U_):
    res = []
    for f in U_:
        f = torch.mean(f, 0, False)
        res.append(f)
    return torch.cat([x.float() for x in res])


class scattering(nn.Module):
    def __init__(self, J, L, V, d_f, K, d, R, lamb_max):
        super(scattering, self).__init__()

        # -- graph parameters
        self.n_node = V
        self.n_atom_features = d_f

        # -- filter parameters
        self.K = K
        self.d = d
        self.J = J
        self.R = R
        self.lamb_max = lamb_max
        self.filters = kernel(K=1, R=3, d=[0.5, 0.5], J=8, lamb_max=2)

        # -- scattering parameters
        self.L = L

        # -- pooling
        self.pooling = pooling

    def compute_spectrum(self, W):
        """
            Computes eigenvalues of normalized graph Laplacian.
        :param W: tensor of graph adjacency matrices.
        :return: eigenvalues of normalized graph Laplacian
        """

        # -- computing Laplacian
        D = torch.diag(W.sum(1))
        L = D - W

        # -- normalize Laplacian
        diag = W.sum(1)
        dhalf = torch.diag_embed(1. / torch.sqrt(torch.max(torch.ones(diag.size()), diag)))
        L = torch.matmul(torch.matmul(dhalf, L), dhalf)

        # -- eig decomposition
        E, V = torch.symeig(L, eigenvectors=True)
        return abs(E), V

    def filtering_matrices(self, W):
        """
            Compute filtering matrices (frames) for spectral filters
        :return: a collection of filtering matrices of each wavelet kernel and the scaling function in the filter-bank.
        """

        filter_matrices = []
        E, V = self.compute_spectrum(W)

        # -- scaling frame
        filter_matrices.append(V @ torch.diag(self.filters.scaling(E)) @ V.T)

        # -- wavelet frame
        for j in range(2, self.J + 1):
            filter_matrices.append(V @ torch.diag(self.filters.wavelet(E, j)) @ V.T)

        return torch.stack(filter_matrices)

    def forward(self, W, f):
        """
            Perform wavelet scattering transform
        :param W: tensor of graph adjacency matrices.
        :param f: tensor of graph signal vectors.
        :return: wavelet scattering coefficients
        """

        # -- filtering matrices
        g = self.filtering_matrices(W)

        # --
        U_ = [f]

        # -- zero-th layer
        S = self.pooling(U_)  # S_(0,1)

        for l in range(self.L):
            U = U_.copy()  # put former res in U
            U_ = []

            for f_ in U:
                for g_j in g:
                    U_.append(abs(g_j @ f_))

            # -- append scattering feature S_(l,i)
            S = torch.cat((S, self.pooling(U_)))
        return S


# -- initialize scattering function
scat = scattering(L=2, V=9, d_f=5, K=1, R=3, d=[0.5, 0.5], J=8, lamb_max=2)

# -- load data
train_data = MolecularDataset(N=5000, train=True)
test_data = MolecularDataset(N=1000, train=False)

# -- Compute scattering feature maps
train_zg = torch.stack(list(map(scat.forward, train_data.adjs, train_data.sigs)))
test_zg = torch.stack(list(map(scat.forward, test_data.adjs, test_data.sigs)))

# -- PCA projection
train_zg = train_zg - torch.mean(train_zg, dim=0)  # mean centering
pca = PCA(n_components=2)
pca.fit(train_zg)
train_zg = pca.transform(train_zg)


test_zg = test_zg - torch.mean(test_zg, dim=0)
pca = PCA(n_components=2)
pca.fit(test_zg)
test_zg = pca.transform(test_zg)

# -- plot feature space

H = train_data.prop_2
plt.scatter(train_zg[:, 0], train_zg[:, 1], c=H)
plt.show()


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


train_dataset = MyDataset(train_data.adjs, torch.DoubleTensor(train_zg), train_data.prop)
test_dataset = MyDataset(test_data.adjs, torch.DoubleTensor(test_zg), test_data.prop)
train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True, drop_last=False, num_workers=0)


class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        # -- initialize weight
        self.weight = Parameter(torch.FloatTensor(2, 2))
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, A, H):
        # -- GCN propagation rule
        A_hat_np = A.numpy() + np.identity(n=A.shape[0])
        D_hat_np = np.squeeze(np.sum(np.array(A_hat_np), axis=1))
        D_hat_inv_sqrt_np = np.diag(np.power(D_hat_np, -1 / 2))
        A_norm = torch.from_numpy(np.dot(np.dot(D_hat_inv_sqrt_np, A_hat_np), D_hat_inv_sqrt_np))
        HW = torch.mm(torch.stack([H.float() for _ in range(9)], 0), self.weight)
        AHW = torch.mm(A_norm.float(), HW)
        # -- non-linearity
        return torch.sum(F.relu(AHW), 0)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # -- initialize layers
        self.gc = GCN()
        self.fc = torch.nn.Linear(2, 1)

    def forward(self, A, h0):
        x = self.gc(A, h0)
        x = self.fc(x)
        return x


# -- Initialize the model, loss function, and the optimizer
model = MyModel()
MyLoss = nn.MSELoss()
MyOptimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# -- update parameters
losses = []
for epoch in range(50):
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

    epoch_loss = running_loss / 200
    losses.append(epoch_loss)

train_loss_value = sum(losses) / 50

# -- plot loss

plt.plot(losses)
plt.show()

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
