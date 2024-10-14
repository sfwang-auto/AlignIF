import os
import torch
import random
import torch.utils.data as data
from torch_geometric.data import Data

BB_ATOMS = ["P", "O5'", "C5'", "C4'", "C3'", "O3'", "C1'", "N"]
LETTER_TO_NUM = {'A': 0, 'G': 1, 'C': 2, 'U': 3, 'N': 4, '_': 4}


class RNADataset(data.Dataset):
    def __init__(self, args, split, radius=20, read_all=False):
        super().__init__()
        self.radius = radius

        self.bb_atoms = args.bb_atoms
        self.central_idx = self.bb_atoms.index(args.central_atom)
        self.bb_idx = [BB_ATOMS.index(bb_atom) for bb_atom in self.bb_atoms]
        
        self.processed_dir = 'data/repre_aligned_processed/'
        self.split = torch.load('data/repre_split.pt')[split]
        self.id_to_align = torch.load('data/repre_id_to_align.pt')
        
        self.read_all = read_all
        if read_all:
            self.data = {}
            for id in self.split:
                self.data[id] = torch.load(os.path.join(self.processed_dir, id + '.pt'))
    
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
    
    def __getitem__(self, idx): 
        id = self.split[idx]
        data = self.data[id] if self.read_all else torch.load(os.path.join(self.processed_dir, id + '.pt'))

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


class AlignIFDataset(data.Dataset):
    def __init__(self, args, split, radius=20, read_all=False):
        super().__init__()
        self.radius = radius
        self.n_aligns = args.n_aligns

        self.bb_atoms = args.bb_atoms
        self.central_idx = self.bb_atoms.index(args.central_atom)
        self.bb_idx = [BB_ATOMS.index(bb_atom) for bb_atom in self.bb_atoms]
        
        self.is_test = split == 'test'
        self.processed_dir = 'data/repre_aligned_processed/'
        self.split = torch.load('data/repre_split.pt')[split]
        self.id_to_align = torch.load('data/repre_id_to_align.pt')

        self.read_all = read_all
        if read_all:
            self.data = {}
            for id in self.split:
                self.data[id] = torch.load(os.path.join(self.processed_dir, id + '.pt'))
    
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
    
    def process(self, id, align_id=None):
        data = self.data[id] if self.read_all else torch.load(os.path.join(self.processed_dir, id + '.pt'))

        coords = torch.tensor(data['coords'][:, self.bb_idx]).to(torch.float32)
        central_coords = coords[:, self.central_idx]

        src_idx, tgt_idx = self.radius_neighbor(central_coords)

        edge_index = torch.cat([src_idx[None], tgt_idx[None]], dim=0)

        seq = torch.tensor(
            [LETTER_TO_NUM[res] for res in data['sequence']]
        )
        mask = torch.logical_and(seq < 4, ~torch.isnan(central_coords[:, 0]))

        # notice that two align structures are exchange
        align = None
        if align_id is not None:
            align = data['align'][align_id]['align']

        num_nodes = seq.shape[0]
        return Data(
            seq=seq, mask=mask, 
            num_nodes=num_nodes, edge_index=edge_index, 
            coords=coords, central_coords=central_coords, 
            align=[align]
        )
    
    def __getitem__(self, idx): 
        id = self.split[idx]
        data = self.process(id)
        
        align_ids = self.id_to_align[id]
        random.shuffle(align_ids)
        align_ids = align_ids[:self.n_aligns]

        data_list = [data]
        if random.random() > 0.5 or self.is_test:
            for align_id in align_ids:
                data_list.append(self.process(align_id, id))
        
        for _ in range(self.n_aligns + 1 - len(data_list)):
            data_list.append(self.process(id))
        
        return data_list