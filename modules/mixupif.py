import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.distributions.beta import Beta

from modules.layers import MPNNLayer


class Featurizer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.central_idx = args.bb_atoms.index(args.central_atom)

        node_in_dim = 44
        edge_in_dim = 688
        hidden_dim = args.hidden_dim
        
        self.node_embedding = nn.Linear(node_in_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_in_dim, hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)
    
    def cal_dihedral(self, X, batch, chain_id, eps=1e-6):
        n_res, n_atoms = X.shape[:2]
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
        dihedral = dihedral.reshape(n_res, n_atoms, -1)

        idx = torch.where(batch[:-1] - batch[1:])[0] + 1
        dihedral[idx, 0] = torch.nan
        dihedral[idx - 1, -2:] = torch.nan

        idx = np.where(chain_id[:-1] != chain_id[1:])[0] + 1
        dihedral[idx, 0] = torch.nan
        dihedral[idx - 1, -2:] = torch.nan
        return torch.cat([torch.sin(dihedral), torch.cos(dihedral)], dim=-1).reshape(n_res, -1)
    
    def cal_angle(self, X, batch, chain_id, eps=1e-6):
        n_res, n_atoms = X.shape[:2]
        X = X.reshape(-1, 3)

        dX0 = F.normalize(X[:-2] - X[1:-1], dim=-1)
        dX1 = F.normalize(X[2:] - X[1:-1], dim=-1)

        cosine = (dX0 * dX1).sum(-1)
        sine = torch.sqrt(1 - cosine.pow(2) + eps)
        sine = F.pad(sine, (1, 1), 'constant', torch.nan)
        cosine = F.pad(cosine, (1, 1), 'constant', torch.nan)

        sine = sine.reshape(n_res, n_atoms, -1)
        cosine = cosine.reshape(n_res, n_atoms, -1)

        idx = torch.where(batch[:-1] - batch[1:])[0] + 1
        sine[idx, 0] = torch.nan
        cosine[idx, 0] = torch.nan
        sine[idx - 1, -1] = torch.nan
        cosine[idx - 1, -1] = torch.nan

        idx = np.where(chain_id[:-1] != chain_id[1:])[0] + 1
        sine[idx, 0] = torch.nan
        cosine[idx, 0] = torch.nan
        sine[idx - 1, -1] = torch.nan
        cosine[idx - 1, -1] = torch.nan
        return torch.cat([sine[:, None], cosine[:, None]], dim=-1).reshape(n_res, -1)

    def rbf(self, D, D_min=0., D_max=20., num_rbf=16):
        D_mu = torch.linspace(D_min, D_max, num_rbf, device=D.device)
        for _ in range(len(D.shape)):
            D_mu = D_mu[None]
        D_sigma = (D_max - D_min) / num_rbf
        return torch.exp(
            -((D[..., None] - D_mu) / D_sigma) ** 2
        )
    
    def cal_dist(self, X, central_X, src_idx, tgt_idx, eps=1e-6):
        inter_dist = torch.sqrt(
            (X[src_idx, None] - X[tgt_idx, :, None]).pow(2).sum(-1) + eps
        )
        inter_dist = self.rbf(inter_dist)
        inter_dist = inter_dist.reshape(inter_dist.shape[0], -1)

        n_atoms = X.shape[1]
        X = X[:, torch.arange(n_atoms) != self.central_idx]
        intra_dist = torch.sqrt(
            (X - central_X[:, None]).pow(2).sum(-1) + eps
        )
        intra_dist = intra_dist.log()
        return intra_dist, inter_dist

    def cal_local_system(self, X, batch, chain_id):
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

        idx = np.where(chain_id[:-1] != chain_id[1:])[0] + 1
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
    
    def cal_direct(self, Q, X, central_X, src_idx, tgt_idx):
        tgt_Q = Q[tgt_idx]
        # inter_direct = F.normalize(X[src_idx] - X[tgt_idx], dim=-1)
        # inter_direct = torch.matmul(tgt_Q[:, None].transpose(-1, -2), inter_direct[..., None])
        inter_direct = F.normalize(X[src_idx, None] - X[tgt_idx, :, None], dim=-1)
        inter_direct = torch.matmul(tgt_Q[:, None, None].transpose(-1, -2), inter_direct[..., None])
        inter_direct = inter_direct.reshape(inter_direct.shape[0], -1)

        n_atoms = X.shape[1]
        X = X[:, torch.arange(n_atoms) != self.central_idx]
        intra_direct = F.normalize(X - central_X[:, None], dim=-1)
        intra_direct = torch.matmul(Q[:, None].transpose(-1, -2), intra_direct[..., None])
        intra_direct = intra_direct.reshape(intra_direct.shape[0], -1)
        return intra_direct, inter_direct

    def forward(self, data):
        batch = data.batch
        coords = data.coords
        chain_id = np.concatenate(data.chain_id)
        central_coords = data.central_coords
        src_idx, tgt_idx = data.edge_index[0], data.edge_index[1]

        dihedral = self.cal_dihedral(coords, batch, chain_id)
        angle = self.cal_angle(coords, batch, chain_id)
        Q = self.cal_local_system(central_coords, batch, chain_id)
        orient = self.cal_orient(Q, src_idx, tgt_idx)
        intra_dist, inter_dist = self.cal_dist(coords, central_coords, src_idx, tgt_idx)
        intra_direct, inter_direct = self.cal_direct(Q, coords, central_coords, src_idx, tgt_idx)  

        h_V = torch.cat([dihedral, angle, intra_dist, intra_direct], -1)
        h_E = torch.cat([orient, inter_dist, inter_direct], -1)

        h_V = torch.nan_to_num(h_V)
        h_E = torch.nan_to_num(h_E)
        h_V = self.node_norm(self.node_embedding(h_V))
        h_E = self.edge_norm(self.edge_embedding(h_E))
        return h_V, h_E
    

# class MixUpIF(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         drop_rate = args.drop_rate
#         hidden_dim = args.hidden_dim
#         self.weight_smooth = args.weight_smooth

#         self.featurizer = Featurizer(args)

#         node_in_dim = 44
#         edge_in_dim = 688
#         self.beta = Beta(torch.tensor([args.alpha]), torch.tensor([args.beta]))
#         self.node_embedding = nn.Linear(node_in_dim, hidden_dim)
#         self.edge_embedding = nn.Linear(edge_in_dim, hidden_dim)
#         self.node_norm = nn.LayerNorm(hidden_dim)
#         self.edge_norm = nn.LayerNorm(hidden_dim)

#         self.encoder_layers = nn.ModuleList(
#             [MPNNLayer(hidden_dim, drop_rate=drop_rate, edge_update=True) for _ in range(args.n_encoder_layers)]
#         )
#         self.decoder_layers = nn.ModuleList(
#             [MPNNLayer(hidden_dim, drop_rate=drop_rate) for _ in range(args.n_decoder_layers)]
#         )
#         self.W_out = nn.Linear(hidden_dim, 4)

#         # Initialization
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)

#     def cal_ce_loss(self, seq, mask, logits, aligned_seq, align_weights):
#         seq = seq * mask
#         seq_onehot = nn.functional.one_hot(seq, 4).float()

#         aligned_seq_onehot = nn.functional.one_hot(aligned_seq.clip(0, 3), 4).float()
#         seq_onehot = align_weights * seq_onehot + (1 - align_weights) * aligned_seq_onehot

#         if self.training:
#             seq_onehot = seq_onehot + self.weight_smooth / 4
#             seq_onehot = seq_onehot / seq_onehot.sum(-1, keepdim=True)

#         log_probs = F.log_softmax(logits, dim=-1)
#         loss = -(seq_onehot * log_probs).sum(-1)
#         return torch.sum(loss * mask) / torch.sum(mask)
    
#     def mixup_feature(self, data):
#         h_V0, h_E0 = self.featurizer(data[0])
#         h_V1, h_E1 = self.featurizer(data[1])

#         batch0 = data[0].batch
#         batch1 = data[1].batch
#         edge_idx0 = data[0].edge_index
#         edge_idx1 = data[1].edge_index

#         device = h_V0.device
#         n_nodes = h_V0.shape[0]
#         n_edges = h_E0.shape[0]

#         aligned_h_V = torch.zeros_like(h_V0)
#         aligned_h_E = torch.zeros_like(h_E0)
#         aligned_seq = torch.zeros_like(data[0].seq)
#         aligned_mask = torch.zeros(n_nodes, device=device, dtype=bool)
#         aligned_edge_mask = torch.zeros(n_edges, device=device, dtype=bool)

#         bsz = batch0[-1] + 1
#         for i in range(bsz):
#             align = data[1].align[i][0]
#             if align is None:
#                 continue
#             else:
#                 # two align structures are exchange in dataset.AlignIFDataset.process()
#                 n_align = align.shape[0]
#                 align0 = torch.tensor(align[:, 1] + (batch0 < i).sum().item(), device=device)
#                 align1 = torch.tensor(align[:, 0] + (batch1 < i).sum().item(), device=device)

#                 edge_mask0 = torch.isin(edge_idx0[0], align0) & torch.isin(edge_idx0[1], align0)
#                 edge_mask1 = torch.isin(edge_idx1[0], align1) & torch.isin(edge_idx1[1], align1)
                
#                 mapping0 = torch.zeros(edge_idx0.max().item() + 1, dtype=torch.long, device=device)
#                 mapping1 = torch.zeros(edge_idx1.max().item() + 1, dtype=torch.long, device=device)
#                 mapping0[align0] = torch.arange(n_align, device=device)
#                 mapping1[align1] = torch.arange(n_align, device=device)

#                 aligned_edge_idx0 = mapping0[edge_idx0[:, edge_mask0]]
#                 aligned_edge_idx1 = mapping1[edge_idx1[:, edge_mask1]]

#                 A0 = torch.zeros((n_align, n_align), device=device)
#                 A1 = torch.zeros((n_align, n_align), device=device)
#                 A0[aligned_edge_idx0[1], aligned_edge_idx0[0]] = 1
#                 A1[aligned_edge_idx1[1], aligned_edge_idx1[0]] = 1
#                 A_idx = torch.where(A0 * A1)[::-1]
#                 A_idx = torch.stack(A_idx, dim=0)

#                 reverse_mapping0 = torch.zeros(align0.max().item() + 1, dtype=torch.long, device=device)
#                 reverse_mapping1 = torch.zeros(align1.max().item() + 1, dtype=torch.long, device=device)
#                 reverse_mapping0[torch.arange(n_align)] = align0
#                 reverse_mapping1[torch.arange(n_align)] = align1

#                 aligned_edge_idx0 = reverse_mapping0[A_idx]
#                 aligned_edge_idx1 = reverse_mapping1[A_idx]

#                 aligned_edge_mask0 = (edge_idx0[:, None] == aligned_edge_idx0[..., None]).all(0).any(0)
#                 aligned_edge_mask1 = (edge_idx1[:, None] == aligned_edge_idx1[..., None]).all(0).any(0)

#                 aligned_h_E[aligned_edge_mask0] = h_E1[aligned_edge_mask1]
#                 aligned_edge_mask[aligned_edge_mask0] = True

#                 aligned_h_V[align0] = h_V1[align1]
#                 aligned_seq[align0] = data[1].seq[align1]
#                 aligned_mask[align0] = data[1].mask[align1]
        
#         # beta merge
#         weights = self.beta.sample((bsz, )).to(device)
#         align_weights = weights[batch0]
#         align_weights[~aligned_mask] = 1
#         align_weights_ = align_weights.expand(h_V0.shape).masked_fill(torch.isnan(aligned_h_V), 1)

#         align_edge_weights = weights[batch0[edge_idx0[0]]]
#         align_edge_weights[~aligned_edge_mask] = 1
#         align_edge_weights_ = align_edge_weights.expand(h_E0.shape).masked_fill(torch.isnan(aligned_h_E), 1)

#         h_V = align_weights_ * h_V0 + (1 - align_weights_) * aligned_h_V
#         h_E = align_edge_weights_ * h_E0 + (1 - align_edge_weights_) * aligned_h_E

#         h_V = torch.nan_to_num(h_V)
#         h_E = torch.nan_to_num(h_E)
#         return h_V, h_E, aligned_seq, align_weights
    
#     def forward(self, data):
#         h_V, h_E, aligned_seq, align_weights = self.mixup_feature(data)
#         h_V = self.node_norm(self.node_embedding(h_V))
#         h_E = self.edge_norm(self.edge_embedding(h_E))

#         for layer in self.encoder_layers:
#             h_V, h_E = layer(h_V, h_E, data[0].edge_index)

#         for layer in self.decoder_layers:
#             h_V = layer(h_V, h_E, data[0].edge_index)
        
#         logits = self.W_out(h_V)
#         ce_loss = self.cal_ce_loss(data[0].seq, data[0].mask, logits, aligned_seq, align_weights)
#         return ce_loss
    
#     def infer(self, data):
#         h_V, h_E = self.featurizer(data)
#         h_V = torch.nan_to_num(h_V)
#         h_E = torch.nan_to_num(h_E)
#         h_V = self.node_norm(self.node_embedding(h_V))
#         h_E = self.edge_norm(self.edge_embedding(h_E))

#         for layer in self.encoder_layers:
#             h_V, h_E = layer(h_V, h_E, data.edge_index)

#         for layer in self.decoder_layers:
#             h_V = layer(h_V, h_E, data.edge_index)
        
#         logits = self.W_out(h_V)
#         pred = logits.argmax(-1)

#         return pred, logits


class MixUpIF(nn.Module):
    def __init__(self, args):
        super().__init__()
        drop_rate = args.drop_rate
        hidden_dim = args.hidden_dim
        self.weight_smooth = args.weight_smooth
        self.central_idx = args.bb_atoms.index(args.central_atom)

        self.featurizer = Featurizer(args)

        self.beta = Beta(torch.tensor([args.alpha]), torch.tensor([args.beta]))

        self.encoder_layers = nn.ModuleList(
            [MPNNLayer(hidden_dim, drop_rate=drop_rate, edge_update=True) for _ in range(args.n_encoder_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [MPNNLayer(hidden_dim, drop_rate=drop_rate) for _ in range(args.n_decoder_layers)]
        )
        self.W_out = nn.Linear(hidden_dim, 4)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def cal_ce_loss(self, seq, mask, logits, aligned_seq, align_weights):
        seq = seq * mask
        seq_onehot = nn.functional.one_hot(seq, 4).float()

        aligned_seq_onehot = nn.functional.one_hot(aligned_seq.clip(0, 3), 4).float()
        seq_onehot = align_weights * seq_onehot + (1 - align_weights) * aligned_seq_onehot

        if self.training:
            seq_onehot = seq_onehot + self.weight_smooth / 4
            seq_onehot = seq_onehot / seq_onehot.sum(-1, keepdim=True)

        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(seq_onehot * log_probs).sum(-1)
        return torch.sum(loss * mask) / torch.sum(mask)
    
    def coords_mixup(self, data):
        batch0 = data[0].batch
        batch1 = data[1].batch
        coords = data[0].coords

        device = coords.device

        aligned_coords = torch.ones_like(coords) * torch.nan
        aligned_seq = torch.zeros_like(data[0].seq)
        aligned_mask = torch.zeros(coords.shape[0], device=device, dtype=bool)

        bsz = batch0[-1] + 1
        for i in range(bsz):
            align = data[1].align[i][0]
            if align is None:
                continue
            else:
                # two align structures are exchange in dataset.AlignIFDataset.process()
                align0 = torch.tensor(align[:, 1] + (batch0 < i).sum().item(), device=device)
                align1 = torch.tensor(align[:, 0] + (batch1 < i).sum().item(), device=device)
                
                aligned_coords[align0] = torch.tensor(data[1].aligned_coords[i][0], device=device)
                aligned_seq[align0] = data[1].seq[align1]
                aligned_mask[align0] = data[1].mask[align1]
        
        # beta merge
        weights = self.beta.sample((bsz, 1)).to(device)
        align_weights = weights[batch0]
        align_weights[~aligned_mask] = 1
        align_weights_ = align_weights.expand(coords.shape).masked_fill(torch.isnan(aligned_coords), 1)

        aligned_coords = torch.nan_to_num(aligned_coords)
        coords = align_weights_ * coords + (1 - align_weights_) * aligned_coords
        data[0].coords = coords

        edge_idx = []
        for i in range(bsz):
            central_coords = coords[batch0 == i, self.central_idx]
            src_idx, tgt_idx = radius_neighbor(central_coords)
            edge_idx.append(torch.cat([src_idx[None], tgt_idx[None]], dim=0) + (batch0 < i).sum())
        edge_idx = torch.cat(edge_idx, dim=-1)
        data[0].edge_index = edge_idx
        return data[0], aligned_seq, align_weights.squeeze(-1)
    
    def feature_mixup(self, data, h_V0, h_V1, h_E0, h_E1):
        batch0 = data[0].batch
        batch1 = data[1].batch
        edge_idx0 = data[0].edge_index
        edge_idx1 = data[1].edge_index

        device = h_V0.device

        aligned_h_V = torch.zeros_like(h_V0)
        aligned_h_E = torch.zeros_like(h_E0)
        aligned_seq = torch.zeros_like(data[0].seq)
        aligned_mask = torch.zeros(h_V0.shape[0], device=device, dtype=bool)
        aligned_edge_mask = torch.zeros(h_E0.shape[0], device=device, dtype=bool)

        bsz = batch0[-1] + 1
        for i in range(bsz):
            align = data[1].align[i][0]
            if align is None:
                continue
            else:
                # two align structures are exchange in dataset.AlignIFDataset.process()
                align0 = torch.tensor(align[:, 1] + (batch0 < i).sum().item(), device=device)
                align1 = torch.tensor(align[:, 0] + (batch1 < i).sum().item(), device=device)

                aligned_h_V[align0] = h_V1[align1]
                aligned_seq[align0] = data[1].seq[align1]
                aligned_mask[align0] = data[1].mask[align1]

                edge_mask0 = torch.isin(edge_idx0[0], align0) & torch.isin(edge_idx0[1], align0)
                edge_mask1 = torch.isin(edge_idx1[0], align1) & torch.isin(edge_idx1[1], align1)

                n_align = align.shape[0]
                mapping0 = torch.zeros(edge_idx0.max().item() + 1, dtype=torch.long, device=device)
                mapping1 = torch.zeros(edge_idx1.max().item() + 1, dtype=torch.long, device=device)
                mapping0[align0] = torch.arange(n_align, device=device)
                mapping1[align1] = torch.arange(n_align, device=device)

                aligned_edge_idx0 = mapping0[edge_idx0[:, edge_mask0]]
                aligned_edge_idx1 = mapping1[edge_idx1[:, edge_mask1]]

                A0 = torch.zeros((n_align, n_align), device=device)
                A1 = torch.zeros((n_align, n_align), device=device)
                A0[aligned_edge_idx0[1], aligned_edge_idx0[0]] = 1
                A1[aligned_edge_idx1[1], aligned_edge_idx1[0]] = 1
                A_idx = torch.where(A0 * A1)[::-1]
                A_idx = torch.stack(A_idx, dim=0)

                reverse_mapping0 = torch.zeros(align0.max().item() + 1, dtype=torch.long, device=device)
                reverse_mapping1 = torch.zeros(align1.max().item() + 1, dtype=torch.long, device=device)
                reverse_mapping0[torch.arange(n_align)] = align0
                reverse_mapping1[torch.arange(n_align)] = align1

                aligned_edge_idx0 = reverse_mapping0[A_idx]
                aligned_edge_idx1 = reverse_mapping1[A_idx]

                aligned_edge_mask0 = (edge_idx0[:, None] == aligned_edge_idx0[..., None]).all(0).any(0)
                aligned_edge_mask1 = (edge_idx1[:, None] == aligned_edge_idx1[..., None]).all(0).any(0)

                aligned_h_E[aligned_edge_mask0] = h_E1[aligned_edge_mask1]
                aligned_edge_mask[aligned_edge_mask0] = True
        
        # beta merge
        weights = self.beta.sample((bsz, )).to(device)
        align_weights = weights[batch0]
        align_weights[~aligned_mask] = 1
        align_weights_ = align_weights.expand(h_V0.shape)

        align_edge_weights = weights[batch0[edge_idx0[0]]]
        align_edge_weights[~aligned_edge_mask] = 1
        align_edge_weights_ = align_edge_weights.expand(h_E0.shape)

        h_V = align_weights_ * h_V0 + (1 - align_weights_) * aligned_h_V
        h_E = align_edge_weights_ * h_E0 + (1 - align_edge_weights_) * aligned_h_E

        return h_V, h_E, aligned_seq, align_weights
    
    # mixup coords
    # def forward(self, data):
    #     data, aligned_seq, align_weights = self.coords_mixup(data)
    #     h_V, h_E = self.featurizer(data)

    #     for layer in self.encoder_layers:
    #         h_V, h_E = layer(h_V, h_E, data.edge_index)

    #     for layer in self.decoder_layers:
    #         h_V = layer(h_V, h_E, data.edge_index)
        
    #     logits = self.W_out(h_V)
    #     ce_loss = self.cal_ce_loss(data.seq, data.mask, logits, aligned_seq, align_weights)
    #     return ce_loss

    def forward(self, data):
        h_V0, h_E0 = self.featurizer(data[0])
        h_V1, h_E1 = self.featurizer(data[1])
        
        for layer in self.encoder_layers:
            h_V0, h_E0 = layer(h_V0, h_E0, data[0].edge_index)
            h_V1, h_E1 = layer(h_V1, h_E1, data[1].edge_index)
        
        h_V, h_E, aligned_seq, align_weights = self.feature_mixup(data, h_V0, h_V1, h_E0, h_E1)

        for layer in self.decoder_layers:
            h_V = layer(h_V, h_E, data[0].edge_index)
        
        logits = self.W_out(h_V)
        ce_loss = self.cal_ce_loss(data[0].seq, data[0].mask, logits, aligned_seq, align_weights)
        return ce_loss
    
    def infer(self, data):
        h_V, h_E = self.featurizer(data)

        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, data.edge_index)

        for layer in self.decoder_layers:
            h_V = layer(h_V, h_E, data.edge_index)
        
        logits = self.W_out(h_V)
        pred = logits.argmax(-1)

        return pred, logits


def radius_neighbor(X, r=20, eps=1e-6):
    dist = torch.sqrt(
        (X[:, None] - X[None]).pow(2).sum(-1) + eps
    )

    n = X.shape[0]
    dist[torch.arange(n), torch.arange(n)] = 0
    tgt_idx, src_idx = torch.where(dist < r)
    return src_idx, tgt_idx
