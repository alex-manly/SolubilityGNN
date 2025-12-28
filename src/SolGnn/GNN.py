from typing import Optional
import torch
import torch.nn.functional as F # for ReLU activation
from torch_geometric.nn import GINEConv # includes edge features
from torch_geometric.nn.aggr import SoftmaxAggregation
from torch_geometric.data import Data

class MolGNN(torch.nn.Module):

    def __init__(
            self,
            node_in_dim: int,
            edge_in_dim: int,
            global_in_dim: int,
            hidden_dim: int = 128,
            num_layers: int = 3,
            aggr: Optional = None
    ):
        super().__init__() # instantiate torch.nn.Module properties

        self.node_in = torch.nn.Linear(node_in_dim, hidden_dim) # embed node features
        self.edge_in = torch.nn.Linear(edge_in_dim, hidden_dim) # embed edge features
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            nn_mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(
                GINEConv(nn_mlp, edge_dim = hidden_dim)
            )

        # we also need to build a MLP for global attributes
        self.global_mlp = torch.nn.Sequential(
            torch.nn.Linear(global_in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )

        if aggr is None:
            aggr = SoftmaxAggregation(learn = True)
        self.aggr = aggr

        self.head = torch.nn.Sequential( # MLP to bring us down to a scalar
            torch.nn.Linear(hidden_dim*2, 32), # node and global feats will be concatenated here
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )

    def forward(self, data: Data):
        x = data.x.float()
        edge_index = data.edge_index
        edge_attr = data.edge_attr.float()
        g = data.global_feat.float()
        batch = data.batch

        x = F.relu(self.node_in(x)) # embed, apply activation to input node features
        edge_attr = F.relu(self.edge_in(edge_attr))

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr) # message passing
            x = F.relu(x)

        graph_embedding = self.aggr(x, index=batch)
        global_embedding = self.global_mlp(g)

        out = self.head(
            torch.cat([graph_embedding, global_embedding], dim = -1)
        )

        return out.view(-1)
