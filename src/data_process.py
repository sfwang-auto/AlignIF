import os
import torch
import pickle
import random
import subprocess
import numpy as np
from Bio import PDB
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.data import Data


RADIUS = 20
TMCUT = 0.45
NUM_RBF = 16
RBF_MAX = 20
USALIGN_PATH = './USalign/USalign'
BB_ATOMS = ["P", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N", "C"]    


def align(args, data, pdb_lines):
    out_pdb_file = os.path.join(args.output_dir, args.name, 'data.pdb')
    with open(out_pdb_file, 'w') as f:
        f.write(pdb_lines)
    
    aligns_dict = {}
    print(f"Aligning ...")
    chain_seq_ids = [chain_id[0] + str(seq_id).rjust(4) for chain_id, seq_id in zip(data['chain_ids'], data['seq_ids'])]
    for file in tqdm(os.listdir(args.pdbs_dir)):
        cmd = [
            USALIGN_PATH,
            out_pdb_file, 
            os.path.join(args.pdbs_dir, file), 
            "-ter", "0",
            "-split", "0", 
            "-a", "T",
            "-do", 
        ]
        result = subprocess.run(" ".join(cmd), capture_output=True, shell=True).stdout.decode('utf-8').split('\n')
        tmscore = float(result[17].split()[1])
        if tmscore >= TMCUT and tmscore < 1:
            with open(os.path.join(args.processed_pdbs_dir, f"{file.split('.')[0]}.pkl"), 'rb') as f:
                align_data = pickle.load(f)
            align_chain_seq_ids = [chain_id[0] + str(seq_id).rjust(4) for chain_id, seq_id in zip(align_data['chain_ids'], align_data['seq_ids'])]
            aligns_dict[file.split('.')[0]] = (tmscore, np.array([(chain_seq_ids.index(line[9:14]), align_chain_seq_ids.index(line[25:30])) for line in result[27:-3]]))
    return aligns_dict


def cal_local_system(X):
    p = F.normalize(X[:, 0] - X[:, 1], dim=-1)
    n = F.normalize(X[:, 2] - X[:, 1], dim=-1)
    b = F.normalize(p + n, dim=-1)
    n = torch.cross(p, n)
    n = F.normalize(n, dim=-1)

    Q = torch.stack((b, n, torch.cross(b, n)), dim=-1)
    return Q

def cal_direct(Q, X):
    P_direct = F.normalize(X[:, BB_ATOMS.index("P")] - X[:, BB_ATOMS.index("C4'")], dim=-1)
    P_direct = torch.matmul(Q.transpose(-1, -2), P_direct[..., None]).squeeze(-1)
    N_direct = F.normalize(X[:, BB_ATOMS.index("N")] - X[:, BB_ATOMS.index("C4'")], dim=-1)
    N_direct = torch.matmul(Q.transpose(-1, -2), N_direct[..., None]).squeeze(-1)
    return torch.cat([P_direct, N_direct], dim=-1)

def quaternions(R):
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

def cal_intra_orient(Q, break_idx):
    forward_R = torch.matmul(Q[:-1].transpose(-1, -2), Q[1:])
    backward_R = torch.matmul(Q[1:].transpose(-1, -2), Q[:-1])
    forward_orient = quaternions(forward_R)
    backward_orient = quaternions(backward_R)
    forward_orient = F.pad(forward_orient, (0, 0, 0, 1), 'constant', torch.nan)
    backward_orient = F.pad(backward_orient, (0, 0, 1, 0), 'constant', torch.nan)
    forward_orient[break_idx - 1] = torch.nan
    backward_orient[break_idx] = torch.nan
    intra_orient = torch.cat([forward_orient, backward_orient], dim=-1)
    return intra_orient

def cal_intra_orient(Q, break_idx):
    forward_R = torch.matmul(Q[:-1].transpose(-1, -2), Q[1:])
    backward_R = torch.matmul(Q[1:].transpose(-1, -2), Q[:-1])
    forward_orient = quaternions(forward_R)
    backward_orient = quaternions(backward_R)
    forward_orient = F.pad(forward_orient, (0, 0, 0, 1), 'constant', torch.nan)
    backward_orient = F.pad(backward_orient, (0, 0, 1, 0), 'constant', torch.nan)
    forward_orient[break_idx - 1] = torch.nan
    backward_orient[break_idx] = torch.nan
    intra_orient = torch.cat([forward_orient, backward_orient], dim=-1)
    return intra_orient

def cal_single_dihedral(v1, v2, v3, eps=1e-6):
    cross1 = F.normalize(torch.cross(v1, v2), dim=-1)
    cross2 = F.normalize(torch.cross(v2, v3), dim=-1)
    dihedral = torch.arccos(
        (cross1 * cross2).sum(-1).clip(-1 + eps, 1 - eps)
    ) * torch.sign((cross2 * v1).sum(-1))
    return dihedral

def cal_dihedral(X_chain, X_sugar, X_base, X_O2p, break_idx):
    n_atoms = X_chain.shape[1]
    X_chain_ = X_chain.reshape(-1, 3)
    dX = X_chain_[1:] - X_chain_[:-1]
    U = F.normalize(dX, dim=-1)

    chain_dihedral = cal_single_dihedral(U[:-2], U[1:-1], U[2:])
    chain_dihedral = F.pad(chain_dihedral, (1, 2), 'constant', torch.nan)
    chain_dihedral = chain_dihedral.reshape(-1, n_atoms)
    chain_dihedral[break_idx, 0] = torch.nan
    chain_dihedral[break_idx - 1, -2:] = torch.nan

    # C2'-C1'-O4'-C4'
    U0 = F.normalize(X_sugar[:, 3] - X_sugar[:, 2], dim=-1)
    U1 = F.normalize(X_sugar[:, 4] - X_sugar[:, 3], dim=-1)
    U2 = F.normalize(X_sugar[:, 0] - X_sugar[:, 4], dim=-1)
    sugar_dihedral1 = cal_single_dihedral(U0, U1, U2)[:, None]

    # O4'-C1'-C2'-O2'
    U0 = F.normalize(X_sugar[:, 3] - X_sugar[:, 4], dim=-1)
    U1 = F.normalize(X_sugar[:, 2] - X_sugar[:, 3], dim=-1)
    U2 = F.normalize(X_O2p - X_sugar[:, 2], dim=-1)
    sugar_dihedral2 = cal_single_dihedral(U0, U1, U2)[:, None]

    # C1'-O4'-N-C
    U0 = F.normalize(X_sugar[:, 3] - X_sugar[:, 4], dim=-1)
    U1 = F.normalize(X_base[:, 0] - X_sugar[:, 3], dim=-1)
    U2 = F.normalize(X_base[:, 1] - X_base[:, 0], dim=-1)
    base_dihedral = cal_single_dihedral(U0, U1, U2)[:, None]

    dihedral = torch.cat([chain_dihedral, sugar_dihedral1, sugar_dihedral2, base_dihedral], dim=-1)
    return torch.cat([torch.cos(dihedral), torch.sin(dihedral)], dim=-1)

def radius_neighbor(X, eps=1e-6):
    n = X.shape[0]
    dist = torch.sqrt(
        (X[:, None] - X[None]).pow(2).sum(-1) + eps
    )
    dist[torch.arange(n), torch.arange(n)] = 0
    tgt_idx, src_idx = torch.where(dist < RADIUS)
    return src_idx, tgt_idx

def rbf(D, num_rbf, rbf_max, rbf_min=0.):
    D_mu = torch.linspace(rbf_min, rbf_max, num_rbf, device=D.device)
    for _ in range(len(D.shape)):
        D_mu = D_mu[None]
    D_sigma = (rbf_max - rbf_min) / num_rbf
    rbf = torch.exp(
        -((D[..., None] - D_mu) / D_sigma) ** 2
    )
    return rbf.reshape(D.size()[:-1] + (-1, ))

def cal_dist(X, src_idx, tgt_idx, eps=1e-6):
    C3p_dist = torch.sqrt((X[src_idx, BB_ATOMS.index("C4'")] - X[tgt_idx, BB_ATOMS.index("C4'")]).pow(2).sum(-1) + eps)[:, None]
    N_dist = torch.sqrt((X[src_idx, BB_ATOMS.index("N")] - X[tgt_idx, BB_ATOMS.index("N")]).pow(2).sum(-1) + eps)[:, None]
    C3p_N_dist = torch.sqrt((X[src_idx, BB_ATOMS.index("C4'")] - X[tgt_idx, BB_ATOMS.index("N")]).pow(2).sum(-1) + eps)[:, None]
    C3p_P_dist = torch.sqrt((X[src_idx, BB_ATOMS.index("C4'")] - X[tgt_idx, BB_ATOMS.index("P")]).pow(2).sum(-1) + eps)[:, None]

    dist = torch.cat([C3p_dist, N_dist, C3p_N_dist, C3p_P_dist], dim=-1)
    return rbf(dist, NUM_RBF, RBF_MAX)

def cal_inter_orient(Q, src_idx, tgt_idx):
    src_Q = Q[src_idx]
    tgt_Q = Q[tgt_idx]
    R = torch.matmul(tgt_Q.transpose(-1, -2), src_Q)
    return quaternions(R)

def graph_construct(data):
    chain_ids = np.array(data['chain_ids'])
    break_idx = np.where(chain_ids[:-1] != chain_ids[1:])[0] + 1
    coords = torch.tensor(data['coords'], dtype=torch.float32)

    O2p_coords = coords[:, BB_ATOMS.index("O2'")]
    sys_coords = coords[:, [BB_ATOMS.index(atom) for atom in ["O4'", "C4'", "C3'"]]]
    base_coords = coords[:, [BB_ATOMS.index(atom) for atom in ["N", "C"]]]
    chain_coords = coords[:, [BB_ATOMS.index(atom) for atom in ["P", "O5'", "C5'", "C4'", "C3'", "O3'"]]]
    sugar_coords = coords[:, [BB_ATOMS.index(atom) for atom in ["C4'", "C3'", "C2'", "C1'", "O4'"]]]

    # process
    Q = cal_local_system(sys_coords)
    direct = cal_direct(Q, coords)
    intra_orient = cal_intra_orient(Q, break_idx)
    dihedral = cal_dihedral(chain_coords, sugar_coords, base_coords, O2p_coords, break_idx)
    h_V = torch.cat([direct, intra_orient, dihedral], -1)

    # remove gap
    central_coords = coords[:, BB_ATOMS.index("C4'")]
    mask = ~torch.isnan(central_coords).any(-1)

    Q = Q[mask]
    h_V = h_V[mask]
    coords = coords[mask]
    central_coords = central_coords[mask]

    src_idx, tgt_idx = radius_neighbor(central_coords)
    edge_index = torch.stack([src_idx, tgt_idx], dim=0)

    dist = cal_dist(coords, src_idx, tgt_idx)
    inter_orient = cal_inter_orient(Q, src_idx, tgt_idx)
    h_E = torch.cat([dist, inter_orient], -1)

    h_V = torch.nan_to_num(h_V)
    h_E = torch.nan_to_num(h_E)
    return Data(h_V=h_V, h_E=h_E, mask=mask, edge_index=edge_index, num_nodes=h_V.shape[0])

def process(args, device):
    if args.input_structure_path.endswith('.cif'): parser = PDB.FastMMCIFParser(auth_residues=False, QUIET=True)
    elif args.input_structure_path.endswith('.pdb'): parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("", args.input_structure_path)[0]
    
    serial = 0
    n_chains = 0
    pdb_lines = ""
    seq_ids = []
    chain_ids = []
    coords_list = []
    chain_id_mapping = {}
    for chain in structure:
        if chain.id not in chain_id_mapping:
            chain_id_mapping[chain.id] = chr(ord('A') + n_chains)
            n_chains += 1
        chain_id = chain_id_mapping[chain.id]
        
        for residue in chain: 
            if residue.id[0] == ' ':
                seq_id = residue.id[1]
                if len(chain_ids) and chain_id == chain_ids[-1] and seq_id - seq_ids[-1] > 1:
                    n_pad = seq_id - seq_ids[-1] - 1
                    chain_ids += [chain_id] * n_pad
                    seq_ids += [seq_ids[-1] + i + 1 for i in range(n_pad)]
                    coords_list.append(np.ones((n_pad, len(BB_ATOMS), 3)) * np.nan)
                
                seq_ids.append(seq_id)
                chain_ids.append(chain_id)
                
                resname = residue.resname
                coords = np.ones((len(BB_ATOMS), 3)) * np.nan
                for atom in residue:
                    coord = atom.coord
                    atom_name = atom.id
                    if atom_name in BB_ATOMS[:-2]: coords[BB_ATOMS.index(atom_name)] = coord
                    elif (atom_name == 'N1' and resname in ['C', 'U']) or (atom_name == 'N9' and resname in ['A', 'G']): coords[-2] = coord
                    elif (atom_name == 'C2' and resname in ['C', 'U']) or (atom_name == 'C4' and resname in ['A', 'G']): coords[-1] = coord

                    serial += 1
                    if len(atom_name) <= 3: atom_name = ' ' + atom_name.ljust(3)
                    pdb_lines += (
                        'ATOM  ' + str(serial).rjust(5) + ' ' + atom_name + atom.altloc + resname.rjust(3) + ' ' + chain_id + 
                        str(seq_id).rjust(4) + residue.id[-1] + '   ' + f"{coord[0]:.3f}".rjust(8) + f"{coord[1]:.3f}".rjust(8) + 
                        f"{coord[2]:.3f}".rjust(8) + f"{atom.occupancy:.2f}".rjust(6) + f"{atom.bfactor:.2f}".rjust(6) + ' ' * 11 + atom.element + '  \n'
                    )
                coords_list.append(coords[None])
        pdb_lines += "TER\n"
    pdb_lines += "END\n"

    target_data = {
        'seq_ids': seq_ids, 
        'chain_ids': chain_ids,
        'chain_id_mapping': chain_id_mapping, 
        'coords': np.concatenate(coords_list, 0), 
    }

    aligns_dict = align(args, target_data, pdb_lines) if args.use_msta else None

    msta_data_list = []
    target_data = graph_construct(target_data)
    target_data.msta_mask = torch.zeros((target_data.h_V.shape[0], args.max_n_aligns), dtype=bool)
    target_data.msta_edge_mask = torch.zeros((target_data.h_E.shape[0], args.max_n_aligns), dtype=bool)
    if args.use_msta:
        msta_ids = [key for key in aligns_dict.keys() if aligns_dict[key][0] >= 0.45]
        random.shuffle(msta_ids)
        msta_ids = msta_ids[:args.max_n_aligns]

        for i, msta_id in enumerate(msta_ids):
            with open(os.path.join(args.processed_pdbs_dir, f"{msta_id}.pkl"), 'rb') as f:
                msta_data =pickle.load(f)
            msta_data = graph_construct(msta_data)

            # mapping real (pdb) indices to data indices
            target_mapping = -torch.ones(target_data.mask.shape[0], dtype=torch.int64)
            msta_mapping = -torch.ones(msta_data.mask.shape[0], dtype=torch.int64) 
            target_mapping[target_data.mask] = torch.arange(target_data.h_V.shape[0])
            msta_mapping[msta_data.mask] = torch.arange(msta_data.h_V.shape[0])

            # mapping align_idx to data indices
            align_idx = torch.tensor(aligns_dict[msta_id][1])
            target_align_idx = target_mapping[align_idx[:, 0]]
            msta_align_idx = msta_mapping[align_idx[:, 1]]
            align_mask = (target_align_idx != -1) * (msta_align_idx != -1)  # to avoid aligned residues not in data
            target_align_idx = target_align_idx[align_mask]
            msta_align_idx = msta_align_idx[align_mask]

            target_data.msta_mask[target_align_idx, i] = True
            msta_data.msta_mask = torch.zeros(msta_data.h_V.shape[0], dtype=bool)
            msta_data.msta_mask[msta_align_idx] = True

            # align edges
            # extract aligned edges
            target_edge_mask = torch.isin(target_data.edge_index, target_align_idx).all(0)
            msta_edge_mask = torch.isin(msta_data.edge_index, msta_align_idx).all(0)

            # mapping edge_index to torch.arange(n_aligned_res)
            n_aligned_res = target_align_idx.shape[0]
            target_edge_mapping = -torch.ones(target_data.h_V.shape[0], dtype=torch.int64)
            target_edge_mapping[target_align_idx] = torch.arange(n_aligned_res)
            msta_edge_mapping = -torch.ones(msta_data.h_V.shape[0], dtype=torch.int64)
            msta_edge_mapping[msta_align_idx] = torch.arange(n_aligned_res)
            target_edge_index = target_edge_mapping[target_data.edge_index[:, target_edge_mask]]
            msta_edge_index = msta_edge_mapping[msta_data.edge_index[:, msta_edge_mask]]

            # align edges on a metrix
            target_A = torch.zeros((n_aligned_res, n_aligned_res), dtype=bool)
            msta_A = torch.zeros((n_aligned_res, n_aligned_res), dtype=bool)
            target_A[target_edge_index[0], target_edge_index[1]] = True
            msta_A[msta_edge_index[0], msta_edge_index[1]] = True
            A_idx = torch.stack(torch.where(target_A * msta_A)[::-1], dim=0)

            # mapping torch.arange(n_aligned_res) to data indices
            target_edge_index = target_align_idx[A_idx]
            msta_edge_index = msta_align_idx[A_idx]
            
            target_data.msta_edge_mask[:, i] = (target_data.edge_index[:, None] == target_edge_index[..., None]).all(0).any(0)
            msta_data.msta_edge_mask = (msta_data.edge_index[:, None] == msta_edge_index[..., None]).all(0).any(0)
            
            msta_data_list.append(msta_data)
    data_list = [target_data] + msta_data_list + [
        Data(
            num_nodes=0, 
            mask=torch.zeros(0, dtype=bool), 
            msta_mask=torch.zeros(0, dtype=bool), 
            msta_edge_mask=torch.zeros(0, dtype=bool), 
            h_V=torch.empty(0, target_data.h_V.shape[-1]), 
            h_E=torch.empty(0, target_data.h_E.shape[-1]), 
            edge_index=torch.zeros((2, 0), dtype=torch.int64), 
        ) for _ in range(args.max_n_aligns - len(msta_data_list))
    ]

    data_list = [data_list_i for data_list_i in data_list if data_list_i.num_nodes > 0]
    max_num_nodes = max(data_list_i.num_nodes for data_list_i in data_list)
    max_num_edges = max(data_list_i.h_E.shape[0] for data_list_i in data_list)
    
    n_mstas = len(data_list)
    node_mask = torch.zeros((n_mstas, max_num_nodes), dtype=bool)
    edge_mask = torch.zeros((n_mstas, max_num_edges), dtype=bool)
    h_V = torch.zeros(n_mstas, max_num_nodes, data_list[0].h_V.shape[-1])
    h_E = torch.zeros(n_mstas, max_num_edges, data_list[0].h_E.shape[-1])
    edge_index = torch.zeros((n_mstas, 2, max_num_edges), dtype=torch.int64)
    msta_mask = torch.zeros((n_mstas - 1, max_num_nodes), dtype=bool)
    msta_edge_mask = torch.zeros((n_mstas - 1, max_num_edges), dtype=bool)
    for i, data in enumerate(data_list):
        n_nodes = data.h_V.shape[0]
        n_edges = data.h_E.shape[0]
        h_V[i, :n_nodes] = data.h_V
        h_E[i, :n_edges] = data.h_E
        node_mask[i, :n_nodes] = True
        edge_mask[i, :n_edges] = True
        edge_index[i, :, :n_edges] = data.edge_index
        if i > 0:
            msta_mask[i - 1, :n_nodes] = data.msta_mask
            msta_edge_mask[i - 1, :n_edges] = data.msta_edge_mask
    target_msta_mask = torch.cat([data_list[0].msta_mask.T[:n_mstas - 1], torch.zeros((n_mstas - 1, max_num_nodes - data_list[0].num_nodes), dtype=bool)], dim=-1)
    target_msta_edge_mask = torch.cat([data_list[0].msta_edge_mask.T[:n_mstas - 1], torch.zeros((n_mstas - 1, max_num_edges - data_list[0].num_edges), dtype=bool)], dim=-1)

    return {
        'h_V': h_V.to(device), 
        'h_E': h_E.to(device), 
        'edge_index': edge_index.to(device), 
        'node_mask': node_mask.to(device), 
        'edge_mask': edge_mask.to(device), 
        'msta_mask': msta_mask.to(device), 
        'msta_edge_mask': msta_edge_mask.to(device), 
        'target_msta_mask': target_msta_mask.to(device), 
        'target_msta_edge_mask': target_msta_edge_mask.to(device), 
        'num_nodes': target_data.num_nodes, 
    }