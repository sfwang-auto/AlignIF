import math
import torch
from torch import nn
from torch_scatter import scatter_sum


class Featurizer(nn.Module):
    def __init__(self, hidden_dim, node_in_dim=32, edge_in_dim=68):
        super().__init__()
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.node_embedding = nn.Linear(node_in_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_in_dim, hidden_dim)
    
    def forward(self, h_V, h_E):
        h_V = self.node_norm(self.node_embedding(h_V))
        h_E = self.edge_norm(self.edge_embedding(h_E))
        return h_V, h_E


class MPNNLayer(nn.Module):
    def __init__(self, num_hidden, drop_rate=0.1, update_edge=False, autoregression=False):
        super().__init__()
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(drop_rate)

        # node
        self.node_norm0 = nn.LayerNorm(num_hidden)
        self.node_norm1 = nn.LayerNorm(num_hidden)
        node_in_dim = num_hidden * 4 if autoregression else num_hidden * 3
        self.node_mlp = nn.Sequential(
            nn.Linear(node_in_dim, num_hidden), 
            self.act, 
            nn.Linear(num_hidden, num_hidden), 
            self.act, 
            nn.Linear(num_hidden, num_hidden)
        )
        self.node_dense = nn.Sequential(
            nn.Linear(num_hidden, num_hidden * 4),
            self.act,
            nn.Linear(num_hidden * 4, num_hidden)
        )

        # edge
        self.update_edge = update_edge
        if update_edge:
            self.edge_norm0 = nn.LayerNorm(num_hidden)
            self.edge_norm1 = nn.LayerNorm(num_hidden)
            self.edge_mlp = nn.Sequential(
                nn.Linear(num_hidden * 3, num_hidden), 
                self.act, 
                nn.Linear(num_hidden, num_hidden), 
                self.act, 
                nn.Linear(num_hidden, num_hidden)
            )
            self.edge_dense = nn.Sequential(
                nn.Linear(num_hidden, num_hidden*4),
                self.act,
                nn.Linear(num_hidden*4, num_hidden)
            )

    def forward(self, h_V, h_E, edge_idx, encoder_h_V=None, order=None):
        src_idx, tgt_idx = edge_idx
        h_EV = torch.cat([h_E, h_V[..., src_idx, :], h_V[..., tgt_idx, :]], dim=-1)

        if encoder_h_V is not None:
            mask = (src_idx < tgt_idx)[:, None] if order is None else (order[src_idx] < order[tgt_idx])[:, None]
            encoder_h_EV = torch.cat([h_E, encoder_h_V[..., src_idx, :], encoder_h_V[..., tgt_idx, :]], dim=-1)
            while len(mask.shape) < len(h_V.shape): mask = mask[None]
            h_EV = mask * h_EV + (~mask) * encoder_h_EV
            
        h_message = self.node_mlp(h_EV)
        scatter_mask = scatter_sum(torch.ones(h_E.shape[-2], device=h_V.device), tgt_idx) > 0
        dh = scatter_sum(
            h_message, tgt_idx, dim=-2
        )[..., scatter_mask, :] / scatter_sum(torch.ones_like(h_message), tgt_idx, dim=-2)[..., scatter_mask, :]
        h_V = self.node_norm0(h_V + self.dropout(dh))
        dh = self.node_dense(h_V)
        h_V = self.node_norm1(h_V + self.dropout(dh))

        if self.update_edge:
            h_EV = torch.cat([h_E, h_V[src_idx], h_V[tgt_idx]], dim=-1)
            dh = self.edge_mlp(h_EV)
            h_E = self.edge_norm0(h_E + self.dropout(dh))
            dh = self.edge_dense(h_E)
            h_E = self.edge_norm1(h_E + self.dropout(dh))  
            return h_V, h_E
        return h_V


class MStAAttention(nn.Module):
    def __init__(self, num_hidden, c, drop_rate=0.1):
        super().__init__()
        self.c = c
        self.n_heads = num_hidden // c
        self.norm = nn.LayerNorm(num_hidden)
        self.dropout = nn.Dropout(drop_rate)
        self.o_lin = nn.Linear(num_hidden, num_hidden)
        self.q_lin = nn.Linear(num_hidden, num_hidden, bias=None)
        self.k_lin = nn.Linear(num_hidden, num_hidden, bias=None)
        self.v_lin = nn.Linear(num_hidden, num_hidden, bias=None)

    def forward(self, h, target_msta_mask, msta_mask):
        if msta_mask.shape[0] > 0:
            msta_h = torch.zeros((msta_mask.shape[0], h.shape[1], h.shape[-1]), device=h.device)
            msta_h[target_msta_mask] = h[1:][msta_mask]
            attn_h = torch.cat([h[:1, target_msta_mask.any(0)], msta_h[:, target_msta_mask.any(0)]], dim=0)

            inverse_mask = torch.cat([target_msta_mask.any(0)[None], target_msta_mask], dim=0)
            attn_mask = torch.cat([
                torch.ones((1, attn_h.shape[1]), dtype=bool, device=h.device), target_msta_mask[:, target_msta_mask.any(0)]
            ], dim=0)

            n_samples, n_attn = attn_h.shape[0], attn_h.shape[1]
            q = self.q_lin(attn_h).reshape(n_samples, n_attn, self.n_heads, self.c)
            k = self.k_lin(attn_h).reshape(n_samples, n_attn, self.n_heads, self.c)
            v = self.v_lin(attn_h).reshape(n_samples, n_attn, self.n_heads, self.c)

            att_map = (q[:, None] * k[None]).sum(-1) / math.sqrt(self.c)
            att_map = att_map - (~attn_mask)[None, ..., None] * 1e9
            att_map = torch.softmax(att_map, dim=1)
            attn_o = (att_map[..., None] * v[None]).sum(1).reshape(n_samples, n_attn, -1)
    
            o = self.v_lin(h)
            o[inverse_mask] = attn_o[attn_mask]
        else:
            o = self.v_lin(h)

        o = self.o_lin(o)
        h = self.norm(h + self.dropout(o))
        return h


class MStAMPNNLayer(MPNNLayer):
    def __init__(self, num_hidden, c, drop_rate=0.1):
        super().__init__(num_hidden, drop_rate, update_edge=True, autoregression=False)
        self.msta_node_attn = MStAAttention(num_hidden, c, drop_rate)
        self.msta_edge_attn = MStAAttention(num_hidden, c, drop_rate)
    
    def forward(
        self, h_V, h_E, edge_idx, 
        edge_mask, msta_mask, msta_edge_mask, 
        target_msta_mask, target_msta_edge_mask
    ):
        src_idx, tgt_idx = edge_idx[:, 0], edge_idx[:, 1]
        src_h_V = torch.gather(h_V, 1, src_idx[..., None].expand(-1, -1, h_V.shape[-1]))
        tgt_h_V = torch.gather(h_V, 1, tgt_idx[..., None].expand(-1, -1, h_V.shape[-1]))
        h_EV = torch.cat([h_E, src_h_V, tgt_h_V], dim=-1)

        h_message = self.node_mlp(h_EV)
        dh = scatter_sum(
            h_message * edge_mask[..., None], tgt_idx, dim=1
        ) / scatter_sum(torch.ones_like(h_message) * edge_mask[..., None], tgt_idx, dim=1).clip(1)
        h_V = self.node_norm0(h_V + self.dropout(dh))

        h_V = self.msta_node_attn(h_V, target_msta_mask, msta_mask)

        dh = self.node_dense(h_V)
        h_V = self.node_norm1(h_V + self.dropout(dh))

        src_h_V = torch.gather(h_V, 1, src_idx[..., None].expand(-1, -1, h_V.shape[-1]))
        tgt_h_V = torch.gather(h_V, 1, tgt_idx[..., None].expand(-1, -1, h_V.shape[-1]))
        h_EV = torch.cat([h_E, src_h_V, tgt_h_V], dim=-1)
        dh = self.edge_mlp(h_EV)
        h_E = self.edge_norm0(h_E + self.dropout(dh))

        h_E = self.msta_edge_attn(h_E, target_msta_edge_mask, msta_edge_mask)

        dh = self.edge_dense(h_E)
        h_E = self.edge_norm1(h_E + self.dropout(dh))  
        return h_V, h_E


class AlignIF(nn.Module):
    def __init__(self, hidden_dim=128, drop_rate=0.2):
        super().__init__()
        
        self.featurizer = Featurizer(hidden_dim)
        self.W_out = nn.Linear(hidden_dim, 4, bias=False)
        self.W_S = nn.Embedding(5, hidden_dim, padding_idx=4)
        self.encoder_layers = nn.ModuleList(
            [MStAMPNNLayer(hidden_dim, c=32, drop_rate=drop_rate) for _ in range(3)]
        )
        self.decoder_layers = nn.ModuleList(
            [MPNNLayer(hidden_dim, drop_rate=drop_rate, autoregression=True) for _ in range(3)]
        )

    @torch.no_grad()
    def sample(self, data, n_samples, temperature=0.1, subseq=None):
        h_V, h_E, edge_index = data['h_V'], data['h_E'], data['edge_index']
        node_mask, edge_mask = data['node_mask'], data['edge_mask']
        msta_mask, msta_edge_mask = data['msta_mask'], data['msta_edge_mask']
        target_msta_mask, target_msta_edge_mask = data['target_msta_mask'], data['target_msta_edge_mask']


        h_V, h_E = self.featurizer(h_V, h_E)

        for layer in self.encoder_layers:
            h_V, h_E = layer(
                h_V, h_E, edge_index, 
                edge_mask, msta_mask, msta_edge_mask, 
                target_msta_mask, target_msta_edge_mask
            )

        h_V = h_V[0, node_mask[0]][None].expand(n_samples, -1, -1)
        h_E = h_E[0, edge_mask[0]][None].expand(n_samples, -1, -1)
        edge_index = edge_index[0, :, edge_mask[0]]
        src_idx, tgt_idx = edge_index
        
        h_V_stack = [h_V] + [h_V.clone() for _ in self.decoder_layers]

        pred = torch.zeros((n_samples, h_V.shape[-2]), dtype=int, device=h_V.device)
        h_S = torch.zeros((n_samples, h_V.shape[-2], h_V.shape[-1]), device=h_V.device)
        logits_list = torch.zeros((n_samples, h_V.shape[-2], 4), device=h_S.device)

        order = None
        subseq_mask = torch.zeros(h_V.shape[-2], dtype=bool)
        if subseq is not None:
            subseq_mask = subseq != 4
            pred[:, subseq_mask] = subseq[subseq_mask]
            h_S[:, subseq_mask] = self.W_S(pred)[:, subseq_mask]
            order = torch.zeros_like(subseq)
            order[subseq_mask] = torch.arange(subseq_mask.sum(), device=subseq.device)
            order[~subseq_mask] = torch.arange((~subseq_mask).sum(), device=subseq.device) + subseq_mask.sum()
            logits_list[:, subseq_mask, subseq[subseq_mask]] = 1e6

        for i in range(h_V.shape[1]):
            if ~subseq_mask[i]:
                edge_mask = tgt_idx == i
                edge_index_ = edge_index[:, edge_mask]
                h_E_ = torch.cat([h_E, h_S[:, src_idx]], dim=-1)[:, edge_mask]

                for j, layer in enumerate(self.decoder_layers):
                    h_V = layer(h_V_stack[j], h_E_, edge_index_, h_V_stack[0], order=order)
                    h_V_stack[j + 1][:, i] = h_V[:, i]
                
                logits = self.W_out(h_V[:, i])
                samples = torch.multinomial(
                    torch.softmax(logits / temperature, dim=-1), 1
                ).squeeze(-1) if temperature > 0 else logits.argmax(-1)

                pred[:, i] = samples
                h_S[:, i] = self.W_S(samples)
                logits_list[:, i] = logits
        return pred, logits_list