import torch
from torch import nn
import torch.nn.functional as F

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
    
    def cal_dihedral(self, X, batch, eps=1e-6):
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
        return torch.cat([torch.sin(dihedral), torch.cos(dihedral)], dim=-1).reshape(n_res, -1)
    
    def cal_angle(self, X, batch, eps=1e-6):
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
        central_coords = data.central_coords
        src_idx, tgt_idx = data.edge_index[0], data.edge_index[1]

        dihedral = self.cal_dihedral(coords, batch)
        angle = self.cal_angle(coords, batch)
        Q = self.cal_local_system(central_coords, batch)
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
    

class AlignIF(nn.Module):
    def __init__(self, args):
        super().__init__()
        drop_rate = args.drop_rate
        hidden_dim = args.hidden_dim
        self.relax_label = args.relax_label
        self.weight_smooth = args.weight_smooth

        self.featurizer = Featurizer(args)

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

    # def cal_ce_loss(self, align_seq, align_mask, mask, logits):
    #     if self.relax_label:
    #         seq = align_seq * mask[None] * align_mask.squeeze(-1)
    #         seq_onehot = nn.functional.one_hot(seq, 4).float()
    #         seq_onehot = (seq_onehot * align_mask).sum(0) / align_mask.sum(0)
    def cal_ce_loss(self, align_seq, align_weights, mask, logits):
        if self.relax_label:
            seq_onehot = nn.functional.one_hot(align_seq.clip(0, 3), 4).float()
            seq_onehot = (seq_onehot * align_weights).sum(0)
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
            h_V, h_E = layer(h_V, h_E, edge_idx)
        return h_V, h_E
    
    def msa_encode(self, data):
        h_V_list, h_E_list = [], []
        for datum in data:
            h_V, h_E = self.encode(datum)
            h_V_list.append(h_V)
            h_E_list.append(h_E)
        
        h_V = h_V_list[0]
        h_E = h_E_list[0]
        seq = data[0].seq
        mask = data[0].mask
        batch = data[0].batch
        edge_idx = data[0].edge_index
        
        aligned_h_V_list = []
        aligned_h_E_list = []
        aligned_seq_list = []
        aligned_mask_list = []
        aligned_edge_mask_list = []

        bsz = batch[-1] + 1
        device = h_V.device
        n_nodes = h_V.shape[0]
        n_edges = h_E.shape[0]
        for datum, h_V_i, h_E_i in zip(data[1:], h_V_list[1:], h_E_list[1:]):
            datum_batch = datum.batch
            datum_edge_idx = datum.edge_index
            aligned_h_V = torch.zeros_like(h_V)
            aligned_h_E = torch.zeros_like(h_E)
            aligned_seq = torch.zeros_like(seq)
            aligned_mask = torch.zeros(n_nodes, device=device, dtype=bool)
            aligned_edge_mask = torch.zeros(n_edges, device=device, dtype=bool)
            for i in range(bsz):
                align = datum.align[i][0]
                if align is None:
                    continue
                else:
                    # two align structures are exchange in dataset.AlignIFDataset.process()
                    n_align = align.shape[0]
                    align0 = torch.tensor(align[:, 1] + (batch < i).sum().item(), device=device)
                    align_i = torch.tensor(align[:, 0] + (datum_batch < i).sum().item(), device=device)

                    edge_mask0 = torch.isin(edge_idx[0], align0) & torch.isin(edge_idx[1], align0)
                    edge_mask_i = torch.isin(datum_edge_idx[0], align_i) & torch.isin(datum_edge_idx[1], align_i)
                    
                    mapping0 = torch.zeros(edge_idx.max().item() + 1, dtype=torch.long, device=device)
                    mapping_i = torch.zeros(datum_edge_idx.max().item() + 1, dtype=torch.long, device=device)
                    mapping0[align0] = torch.arange(n_align, device=device)
                    mapping_i[align_i] = torch.arange(n_align, device=device)

                    edge_idx0 = mapping0[edge_idx[:, edge_mask0]]
                    edge_idx_i = mapping_i[datum_edge_idx[:, edge_mask_i]]

                    A0 = torch.zeros((n_align, n_align), device=device)
                    A_i = torch.zeros((n_align, n_align), device=device)
                    A0[edge_idx0[1], edge_idx0[0]] = 1
                    A_i[edge_idx_i[1], edge_idx_i[0]] = 1
                    A_idx = torch.where(A0 * A_i)[::-1]
                    A_idx = torch.stack(A_idx, dim=0)

                    reverse_mapping0 = torch.zeros(align0.max().item() + 1, dtype=torch.long, device=device)
                    reverse_mapping_i = torch.zeros(align_i.max().item() + 1, dtype=torch.long, device=device)
                    reverse_mapping0[torch.arange(n_align)] = align0
                    reverse_mapping_i[torch.arange(n_align)] = align_i

                    aligned_edge_idx = reverse_mapping0[A_idx]
                    edge_idx_i = reverse_mapping_i[A_idx]

                    edge_idx0 = (edge_idx[:, None] == aligned_edge_idx[..., None]).all(0).any(0)
                    edge_idx_i = (datum_edge_idx[:, None] == edge_idx_i[..., None]).all(0).any(0)

                    aligned_h_E[edge_idx0] = h_E_i[edge_idx_i]
                    aligned_edge_mask[edge_idx0] = True

                    aligned_h_V[align0] = h_V_i[align_i]
                    aligned_seq[align0] = datum.seq[align_i]
                    aligned_mask[align0] = datum.mask[align_i]
            aligned_h_V_list.append(aligned_h_V)
            aligned_h_E_list.append(aligned_h_E)
            aligned_seq_list.append(aligned_seq)
            aligned_mask_list.append(aligned_mask)
            aligned_edge_mask_list.append(aligned_edge_mask)
        h_V = torch.stack([h_V] + aligned_h_V_list, dim=0)
        h_E = torch.stack([h_E] + aligned_h_E_list, dim=0)
        aligned_seq = torch.stack([seq] + aligned_seq_list, dim=0)
        aligned_mask = torch.stack([torch.ones(n_nodes, device=device, dtype=bool)] + aligned_mask_list, dim=0)
        aligned_edge_mask = torch.stack([torch.ones(n_edges, device=device, dtype=bool)] + aligned_edge_mask_list, dim=0)
        
        # average merge
        n_structures = h_V.shape[0]
        weights = torch.rand((n_structures, bsz), device=device)
        align_weights = weights[:, batch, None]
        align_weights = (align_weights * aligned_mask[..., None]) / (align_weights * aligned_mask[..., None]).sum(0, keepdim=True)
        align_edge_weights = weights[:, batch[edge_idx[0]], None]
        align_edge_weights = (align_edge_weights * aligned_edge_mask[..., None]) / (align_edge_weights * aligned_edge_mask[..., None]).sum(0, keepdim=True)

        node_align_loss, edge_align_loss = 0, 0
        node_align_counts, edge_align_counts = 0, 0
        for i in range(1, n_structures):
            valid_aligned_mask = mask * aligned_mask[i].squeeze(-1)
            valid_aligned_h_V = h_V[:, valid_aligned_mask]
            valid_aligned_h_E = h_E[:, aligned_edge_mask[i]]
            node_align_loss = node_align_loss + (valid_aligned_h_V[0] - valid_aligned_h_V[1]).pow(2).sum(-1).mean()
            edge_align_loss = edge_align_loss + (valid_aligned_h_E[0] - valid_aligned_h_E[1]).pow(2).sum(-1).mean()
            node_align_counts += valid_aligned_mask.sum()
            edge_align_counts += aligned_edge_mask[i].sum()

        # h_V = (h_V * align_weights).sum(0)
        # h_E = (h_E * align_edge_weights).sum(0)
        h_V = h_V[0]
        h_E = h_E[0]
        return h_V, h_E, aligned_seq, align_weights, node_align_loss, node_align_counts, edge_align_loss, edge_align_counts
    
    def forward(self, data):
        # h_V, h_E, align_seq, align_mask = self.msa_encode(data)
        h_V, h_E, align_seq, align_weights, node_align_loss, node_align_counts, edge_align_loss, edge_align_counts = self.msa_encode(data)

        for layer in self.decoder_layers:
            h_V = layer(h_V, h_E, data[0].edge_index)
        
        logits = self.W_out(h_V)
        # ce_loss = self.cal_ce_loss(align_seq, align_mask, data[0].mask, logits)
        ce_loss = self.cal_ce_loss(align_seq, align_weights, data[0].mask, logits)
        return ce_loss, node_align_loss, node_align_counts, edge_align_loss, edge_align_counts
    
    def infer(self, data):
        h_V, h_E, _, _, _, _, _, _ = self.msa_encode(data)

        for layer in self.decoder_layers:
            h_V = layer(h_V, h_E, data[0].edge_index)
        
        logits = self.W_out(h_V)
        pred = logits.argmax(-1)

        return pred, logits