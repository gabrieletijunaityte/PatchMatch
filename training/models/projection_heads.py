import torch.nn as nn

"""Projection head definitions"""

class ProjectionHeadSimCLR(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, 4 * output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * output_dim, output_dim)
        )

    def forward(self, x):
        return self.projection_head(x)


class ProjectionHeadLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        return self.projection_head(x)


class ProjectionHeadFNN(nn.Module):
    # Wang et al., 2021 https://github.com/wangxu-scu/DRSL
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.projection_head = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                             nn.BatchNorm1d(hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(hidden_dim, hidden_dim),
                                             nn.BatchNorm1d(hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(hidden_dim, output_dim),
                                             nn.BatchNorm1d(output_dim),
                                             nn.ReLU()
                                             )
        self.output_dim = output_dim

    def forward(self, x):
        return self.projection_head(x)
