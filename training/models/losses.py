import numpy as np
import torch
from pytorch_metric_learning.losses import NTXentLoss
from torch import nn
from torch.nn import functional as F

""" Functions defining different losses and their calculations"""

class CosineEmbeddingLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
        self.loss = nn.CosineEmbeddingLoss(margin=self.margin, reduction='mean')

    def forward(self, z_left, z_right, match, **kwargs):

        # Change negative pair labels to -1
        match_label = match.clone()
        match_label[match != 1] = -1

        return self.loss(z_left, z_right, match_label).to(torch.float32)


class NTXentLossIn(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_left, z_right, right_IDs, **kwargs):

        # Get normalised similarity matrix
        cos_sim = F.cosine_similarity(z_left[:, None, :], z_right[None, :, :], dim=-1)
        cos_sim = cos_sim / self.temperature

        # Obtain label matrix
        left_IDs = right_IDs.unsqueeze(1)
        pos_mask = right_IDs == left_IDs

        # List to store individual per row losses
        ntx_losses = []

        for i, row in enumerate(cos_sim):
            ntx_loss_row = 0
            pos_sim = row[pos_mask[i, :]]
            neg_sim = row[~pos_mask[i, :]]
            for pos_sim_val in pos_sim:
                loss_val = pos_sim_val - torch.logsumexp(torch.cat((neg_sim, pos_sim_val.unsqueeze(0)), dim=-1), dim=-1)
                ntx_loss_row += loss_val

            loss_val = - ntx_loss_row
            ntx_losses.append(loss_val)

        # Average losses across rows
        ntx_losses = torch.stack(ntx_losses)
        ntx_loss = torch.mean(ntx_losses).to(torch.float32)
        return ntx_loss


class NTXentLossOut(nn.Module):
    def __init__(self, temperature ):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_left, z_right, right_IDs, **kwargs):

        # Get normalised similarity matrix
        cos_sim = F.cosine_similarity(z_left[:, None, :], z_right[None, :, :], dim=-1)
        cos_sim = cos_sim / self.temperature

        left_IDs = right_IDs.unsqueeze(1)
        pos_mask = right_IDs == left_IDs

        # Obtain label matrix
        ntx_losses = []

        for i, row in enumerate(cos_sim):
            pos_sim = row[pos_mask[i, :]]
            loss_val = - torch.logsumexp(pos_sim, dim=-1) + torch.logsumexp(row, dim=-1)
            ntx_losses.append(loss_val)

        # Average losses across rows
        ntx_losses = torch.stack(ntx_losses)
        ntx_loss = torch.mean(ntx_losses).to(torch.float32)
        return ntx_loss


class NTXentLossFull(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        # https://github.com/KevinMusgrave/pytorch-metric-learning/issues/6
        # https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#ntxentloss
        self.loss = NTXentLoss(temperature)

    def forward(self, z_left, z_right, right_IDs, **kwargs):
        embeddings = torch.cat((z_left, z_right))
        labels = torch.cat((right_IDs, right_IDs))
        loss = self.loss(embeddings, labels).to(torch.float32)
        return loss


class ClipLoss(nn.Module):
    # https://www.kaggle.com/code/najkashyap/clip-model-pytorch
    def __init__(self, temperature=None):
        super().__init__()
        self.temperature = nn.Parameter(torch.log(torch.tensor(temperature)))

    def convert_labels(self, right_IDs):
        device = right_IDs.device
        right_IDs = right_IDs.tolist()

        # https://www.w3schools.com/python/ref_dictionary_setdefault.asp
        unique_IDs_map = {}
        unique_IDs = [unique_IDs_map.setdefault(value, idx) for idx, value in enumerate(right_IDs)]

        return torch.tensor(unique_IDs, device=device, requires_grad=False)

    def forward(self, z_left, z_right, right_IDs, **kwargs):

        # Normalise inputs
        z_left = F.normalize(z_left, dim=-1)
        z_right = F.normalize(z_right, dim=-1)

        # Clip temperature to not exceed 100
        temperature =  torch.clamp(self.temperature.exp(), max=100)

        # Get cosine similarity
        dot_product = (z_left @ z_right.T) / temperature

        # Convert labels to 0-n range
        labels = self.convert_labels(right_IDs)

        # Calculate losses per platforms
        loss1 = F.cross_entropy(dot_product, labels)
        loss2 = F.cross_entropy(dot_product.T, labels)

        return ((loss1 + loss2) / 2).to(torch.float32)


class DeepRelationalSimilarityLoss(nn.Module):
    # Wang et al., 2021 https://github.com/wangxu-scu/DRSL
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss(reduction='none')

    def get_label_matrix(self, right_IDs):
        # A priori
        left_IDs = right_IDs.unsqueeze(1)
        pair_matrix = right_IDs == left_IDs
        pair_matrix = pair_matrix.float()

        return pair_matrix

    def forward(self, similarity, right_IDs, **kwargs):

        # Get a priori matrix (labels)
        labels = self.get_label_matrix(right_IDs)

        # Flatten them
        flat_labels = labels.reshape(-1, 1)
        flat_similarity = similarity.reshape(-1, 1)

        # Calculate losses per sample
        losses = self.loss(flat_similarity, flat_labels)

        # Average losses
        loss = losses.sum() / np.sqrt(flat_labels.size(0))
        return loss, losses, labels