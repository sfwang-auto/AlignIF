import torch
from torch import nn
from torch_scatter import scatter_sum


class MPNNLayer(nn.Module):
    def __init__(self, num_hidden, drop_rate=0.1, edge_update=False):
        super().__init__()
        self.dropout = nn.Dropout(drop_rate)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        
        self.act = nn.ReLU()
        self.node_mlp = nn.Sequential(
            nn.Linear(num_hidden * 3, num_hidden, bias=True), 
            self.act, 
            nn.Linear(num_hidden, num_hidden, bias=True), 
            self.act, 
            nn.Linear(num_hidden, num_hidden, bias=True)
        )
        
        self.node_dense = nn.Sequential(
            nn.Linear(num_hidden, num_hidden*4),
            nn.ReLU(),
            nn.Linear(num_hidden*4, num_hidden)
        )

        self.edge_update = edge_update
        if edge_update:
            self.norm3 = nn.LayerNorm(num_hidden)
            self.norm4 = nn.LayerNorm(num_hidden)

            self.edge_mlp = nn.Sequential(
                nn.Linear(num_hidden * 3, num_hidden, bias=True), 
                self.act, 
                nn.Linear(num_hidden, num_hidden, bias=True), 
                self.act, 
                nn.Linear(num_hidden, num_hidden, bias=True)
            )

            self.edge_dense = nn.Sequential(
                nn.Linear(num_hidden, num_hidden*4),
                nn.ReLU(),
                nn.Linear(num_hidden*4, num_hidden)
            )

    def forward(self, h_V, h_E, edge_idx):
        src_idx, tgt_idx = edge_idx[0], edge_idx[1]
        h_EV = torch.cat([h_E, h_V[src_idx], h_V[tgt_idx]], dim=-1)
        h_message = self.node_mlp(h_EV)
        dh = scatter_sum(h_message, tgt_idx, dim=0) / scatter_sum(torch.ones_like(h_message), tgt_idx, dim=0)

        h_V = self.norm1(h_V + self.dropout(dh))
        dh = self.node_dense(h_V)
        h_V = self.norm2(h_V + self.dropout(dh))

        if self.edge_update:
            h_EV = torch.cat([h_E, h_V[src_idx], h_V[tgt_idx]], dim=-1)
            dh = self.edge_mlp(h_EV)
            h_E = self.norm3(h_E + self.dropout(dh))
            dh = self.edge_dense(h_E)
            h_E = self.norm4(h_E + self.dropout(dh))
            return h_V, h_E
        else:
            return h_V