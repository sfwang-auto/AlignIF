import torch
from torch import nn
import torch.nn.functional as F

from modules.layers import MPNNLayer


class Featurizer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.central_idx = args.bb_atoms.index(args.central_atom)

        node_in_dim = 4
        edge_in_dim = 23
        hidden_dim = args.hidden_dim

        self.node_embedding = nn.Linear(node_in_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_in_dim, hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)
    
    def cal_dihedral(self, X, batch, eps=1e-6):
        X = X.reshape(-1, 3)

        dX = X[1:] - X[:-1]
        U = F.normalize(dX, dim=-1)

        cross1 = torch.cross(U[:-2], U[1:-1])
        cross2 = torch.cross(U[1:-1], U[2:])
        cross1 = F.normalize(cross1, dim=-1)
        cross2 = F.normalize(cross2, dim=-1)

        dihedral = torch.arccos(
            (cross1 * cross2).sum(-1).clip(-1 + eps, 1 - eps)
        ) * torch.sign((cross2 * U[:-2]).sum(-1))
        dihedral = F.pad(dihedral, (1, 2), 'constant', torch.nan)

        idx = torch.where(batch[:-1] - batch[1:])[0] + 1
        dihedral[idx] = torch.nan
        dihedral[idx - 1] = torch.nan
        dihedral[idx - 2] = torch.nan
        return torch.cat([torch.sin(dihedral)[:, None], torch.cos(dihedral)[:, None]], dim=-1)
    
    def cal_angle(self, X, batch, eps=1e-6):
        dX0 = F.normalize(X[:-2] - X[1:-1], dim=-1)
        dX1 = F.normalize(X[2:] - X[1:-1], dim=-1)

        cosine = (dX0 * dX1).sum(-1)
        sine = torch.sqrt(1 - cosine.pow(2) + eps)
        sine = F.pad(sine, (1, 1), 'constant', torch.nan)
        cosine = F.pad(cosine, (1, 1), 'constant', torch.nan)

        idx = torch.where(batch[:-1] - batch[1:])[0] + 1
        sine[idx] = torch.nan
        cosine[idx] = torch.nan
        sine[idx - 1] = torch.nan
        cosine[idx - 1] = torch.nan
        return torch.cat([sine[:, None], cosine[:, None]], dim=-1)

    def rbf(self, D, D_min=0., D_max=20., num_rbf=16):
        D_mu = torch.linspace(D_min, D_max, num_rbf, device=D.device)
        for _ in range(len(D.shape)):
            D_mu = D_mu[None]
        D_sigma = (D_max - D_min) / num_rbf
        return torch.exp(
            -((D[..., None] - D_mu) / D_sigma) ** 2
        )
    
    def cal_dist(self, central_X, src_idx, tgt_idx, eps=1e-6):
        dist = torch.sqrt(
            (central_X[src_idx] - central_X[tgt_idx]).pow(2).sum(-1) + eps
        )
        dist = self.rbf(dist)
        return dist

    def cal_local_system(self, X, batch):
        dX = X[1:] - X[:-1]
        u = F.normalize(dX, dim=-1)
        b = u[:-1] - u[1:]
        b = F.normalize(b, dim=-1)
        n = torch.cross(u[:-1], u[1:])
        n = F.normalize(n, dim=-1)

        # Q = [b, n, b ✖️ n]
        Q = torch.stack((b, n, torch.cross(b, n)), dim=-1)
        Q = F.pad(Q, (0, 0, 0, 0, 1, 1), 'constant', torch.nan)

        idx = torch.where(batch[:-1] - batch[1:])[0] + 1
        Q[idx] = torch.nan
        Q[idx - 1] = torch.nan
        return Q
    
    def quaternions(self, R):
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
              Rxx - Ryy - Rzz, 
            - Rxx + Ryy - Rzz, 
            - Rxx - Ryy + Rzz
        ], -1)))
        signs = torch.sign(torch.stack([
            R[:, 2,1] - R[:, 1,2],
            R[:, 0,2] - R[:, 2,0],
            R[:, 1,0] - R[:, 0,1]
        ], -1))
        xyz = signs * magnitudes
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        q = torch.cat((xyz, w), -1)
        q = F.normalize(q, dim=-1)
        return q
    
    def cal_orient(self, Q, src_idx, tgt_idx):
        src_Q = Q[src_idx]
        tgt_Q = Q[tgt_idx]
        R = torch.matmul(tgt_Q.transpose(-1, -2), src_Q)
        return self.quaternions(R)
    
    def cal_direct(self, Q, central_X, src_idx, tgt_idx):
        tgt_Q = Q[tgt_idx]
        direct = F.normalize(central_X[src_idx] - central_X[tgt_idx], dim=-1)
        direct = torch.matmul(tgt_Q.transpose(-1, -2), direct[..., None]).squeeze(-1)
        return direct

    def forward(self, data):
        batch = data.batch
        central_coords = data.central_coords
        src_idx, tgt_idx = data.edge_index[0], data.edge_index[1]

        dihedral = self.cal_dihedral(central_coords, batch)
        angle = self.cal_angle(central_coords, batch)
        Q = self.cal_local_system(central_coords, batch)
        orient = self.cal_orient(Q, src_idx, tgt_idx)
        dist = self.cal_dist(central_coords, src_idx, tgt_idx)
        direct = self.cal_direct(Q, central_coords, src_idx, tgt_idx)  

        h_V = torch.cat([dihedral, angle], -1)
        h_E = torch.cat([orient, dist, direct], -1)

        h_V = torch.nan_to_num(h_V)
        h_E = torch.nan_to_num(h_E)
        h_V = self.node_norm(self.node_embedding(h_V))
        h_E = self.edge_norm(self.edge_embedding(h_E))
        return h_V, h_E
    

class AlignIF(nn.Module):
    def __init__(self, args):
        super().__init__()
        drop_rate = args.drop_rate
        hidden_dim = args.hidden_dim
        self.loose_label = args.loose_label
        self.weight_smooth = args.weight_smooth

        self.featurizer = Featurizer(args)

        self.encoder_layers = nn.ModuleList(
            [MPNNLayer(hidden_dim, drop_rate=drop_rate) for _ in range(args.n_encoder_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [MPNNLayer(hidden_dim, drop_rate=drop_rate) for _ in range(args.n_decoder_layers)]
        )
        self.W_out = nn.Linear(hidden_dim, 4)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def cal_ce_loss(self, align_seq, align_mask, mask, logits):
        if self.loose_label:
            seq = align_seq * mask[None] * align_mask.squeeze(-1)
            seq_onehot = nn.functional.one_hot(seq, 4).float()
            seq_onehot = (seq_onehot * align_mask).sum(0) / align_mask.sum(0)
        else:
            seq = align_seq[0] * mask
            seq_onehot = nn.functional.one_hot(seq, 4).float()

        if self.training:
            seq_onehot = seq_onehot + self.weight_smooth / 4
            seq_onehot = seq_onehot / seq_onehot.sum(-1, keepdim=True)

        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(seq_onehot * log_probs).sum(-1)
        return torch.sum(loss * mask) / torch.sum(mask)

    def encode(self, datum):
        edge_idx = datum.edge_index 
        h_V, h_E = self.featurizer(datum)
        for layer in self.encoder_layers:
            h_V = layer(h_V, h_E, edge_idx)
        return h_V, h_E
    
    def forward(self, data):
        h_V_list, h_E_list = [], []
        for datum in data:
            h_V, h_E = self.encode(datum)
            h_V_list.append(h_V)
            h_E_list.append(h_E)
        
        h_V = h_V_list[0]
        h_E = h_E_list[0]
        seq = data[0].seq
        align_h_V_list = []
        align_seq_list = []
        align_mask_list = []
        batch = data[0].batch
        for datum, h_V_align in zip(data[1:], h_V_list[1:]):
            datum_batch = datum.batch
            align_h_V = torch.zeros_like(h_V)
            align_seq = torch.zeros_like(seq)
            align_mask = torch.zeros(h_V.shape[0], device=h_V.device, dtype=bool)
            for i in range(datum_batch[-1] + 1):
                align = datum.align[i][0]
                if align is None:
                    continue
                else:
                    # two align structures are exchange in dataset.AlignIFDataset.process()
                    align0 = align[:, 1] + (batch < i).sum().item()
                    align_i = align[:, 0] + (datum_batch < i).sum().item()
                    align_h_V[align0] = h_V_align[align_i]
                    align_seq[align0] = datum.seq[align_i]
                    align_mask[align0] = datum.mask[align_i]
            align_h_V_list.append(align_h_V)
            align_seq_list.append(align_seq)
            align_mask_list.append(align_mask)
        h_V = torch.stack([h_V] + align_h_V_list, dim=0)
        align_seq = torch.stack([seq] + align_seq_list, dim=0)
        align_mask = torch.stack([torch.ones(h_V.shape[1], device=h_V.device, dtype=bool)] + align_mask_list, dim=0)[..., None]
        
        # average merge
        h_V = (h_V * align_mask).sum(0) / align_mask.sum(0)

        data = data[0]
        edge_idx = data.edge_index
        for layer in self.decoder_layers:
            h_V = layer(h_V, h_E, edge_idx)
        
        logits = self.W_out(h_V)
        ce_loss = self.cal_ce_loss(align_seq, align_mask, data.mask, logits)
        return ce_loss
    
    def infer(self, data):
        h_V_list, h_E_list = [], []
        for datum in data:
            h_V, h_E = self.encode(datum)
            h_V_list.append(h_V)
            h_E_list.append(h_E)
        
        h_V = h_V_list[0]
        h_E = h_E_list[0]
        align_h_V_list = []
        align_mask_list = []
        batch = data[0].batch
        for datum, h_V_align in zip(data[1:], h_V_list[1:]):
            datum_batch = datum.batch
            align_h_V = torch.zeros_like(h_V)
            align_mask = torch.zeros(h_V.shape[0], device=h_V.device, dtype=bool)
            for i in range(datum_batch[-1] + 1):
                align = datum.align[i][0]
                if align is None:
                    continue
                else:
                    # two align structures are exchange in dataset.AlignIFDataset.process()
                    align0 = align[:, 1] + (batch < i).sum().item()
                    align_i = align[:, 0] + (datum_batch < i).sum().item()
                    align_h_V[align0] = h_V_align[align_i]
                    align_mask[align0] = datum.mask[align_i]
            align_h_V_list.append(align_h_V)
            align_mask_list.append(align_mask)
        h_V = torch.stack([h_V] + align_h_V_list, dim=0)
        align_mask = torch.stack([torch.ones(h_V.shape[1], device=h_V.device, dtype=bool)] + align_mask_list, dim=0)[..., None]
        
        # average merge
        h_V = (h_V * align_mask).sum(0) / align_mask.sum(0)

        data = data[0]
        edge_idx = data.edge_index
        for layer in self.decoder_layers:
            h_V = layer(h_V, h_E, edge_idx)
        
        logits = self.W_out(h_V)
        pred = logits.argmax(-1)

        return pred, logits