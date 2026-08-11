import torch
from torch import nn
from torch.nn import functional as F

"""Functions to calculate similarity"""

class CosSimilarity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z_left, z_right, dim=1, **kwargs):
        sim = F.cosine_similarity(z_left, z_right, dim=dim)
        return sim

class DeepRelationalSimilarity(nn.Module):
    # Based on code provided by Wang et al., 2021 https://github.com/wangxu-scu/DRSL
    def __init__(self, input_dim=600, hidden_dim=1024, output_dim=1):
        super().__init__()

        # Relational NN
        self.RelationNN = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z_left, z_right, dim=-1, **kwargs):
        """Relational similarity calculation"""

        if dim == 1:
            # To obtain full similarity matrix
            relation_score = self.similarity_matrix(z_left, z_right)
        else:
            # To obtain similarity just for the given pairs
            y = torch.cat((z_left, z_right), 1)
            relation_score = self.RelationNN(y)
        return relation_score

    def similarity_matrix(self, z_left, z_right):
        batch_size = z_left.shape[0]

        ni = z_left.size(0)
        di = z_left.size(1)
        nt = z_right.size(0)
        dt = z_right.size(1)
        z_left = z_left.unsqueeze(1).expand(ni, nt, di)
        z_left = z_left.reshape(-1, di)

        z_right = z_right.unsqueeze(0).expand(ni, nt, dt)
        z_right = z_right.reshape(-1, dt)

        y = torch.cat((z_left, z_right), 1)

        relation_score = self.RelationNN(y)
        relation_matrix = relation_score.reshape(batch_size, batch_size)
        return relation_matrix