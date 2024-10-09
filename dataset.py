import os
import torch
from torch import nn
import torch.utils.data as data
from torch_geometric.data import Data

BB_ATOMS = ["P", "O5'", "C5'", "C4'", "C3'", "O3'", "C1'", "N"]
LETTER_TO_NUM = {'A': 0, 'G': 1, 'C': 2, 'U': 3, 'N': 4, '_': 4}


class RNADataset(data.Dataset):
    def __init__(self, args, split, radius=20):
        super().__init__()
        self.radius = radius

        self.bb_atoms = args.bb_atoms
        self.central_idx = self.bb_atoms.index(args.central_atom)
        self.bb_idx = [BB_ATOMS.index(bb_atom) for bb_atom in self.bb_atoms]
        
        self.processed_dir = 'data/repre_aligned_processed/'
        self.split = torch.load('data/repre_split.pt')[split]
    
    def radius_neighbor(self, X, eps=1e-6):
        dist = torch.sqrt(
            (X[:, None] - X[None]).pow(2).sum(-1) + eps
        )

        n = X.shape[0]
        dist[torch.arange(n), torch.arange(n)] = 0
        tgt_idx, src_idx = torch.where(dist < self.radius)
        return src_idx, tgt_idx

    def __len__(self): 
        return len(self.split)
    
    def __getitem__(self, i): 
        data = torch.load(os.path.join(self.processed_dir, self.split[i] + '.pt'))

        coords = torch.tensor(data['coords'][:, self.bb_idx]).to(torch.float32)
        central_coords = coords[:, self.central_idx]

        src_idx, tgt_idx = self.radius_neighbor(central_coords)

        edge_index = torch.cat([src_idx[None], tgt_idx[None]], dim=0)

        seq = torch.tensor(
            [LETTER_TO_NUM[res] for res in data['sequence']]
        )
        mask = torch.logical_and(seq < 4, ~torch.isnan(central_coords[:, 0]))

        num_nodes = seq.shape[0]
        return Data(
            seq=seq, mask=mask, 
            num_nodes=num_nodes, edge_index=edge_index, 
            coords=coords, central_coords=central_coords, 
        )