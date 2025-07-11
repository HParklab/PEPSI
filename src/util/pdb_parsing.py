import numpy as np
import torch
from typing import Tuple, List
from torch_geometric.data import Data
from itertools import groupby
from collections import defaultdict
from Bio.PDB import PDBParser

def calculate_virtual_cb(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
 
    v1 = n - ca 
    v2 = c - ca  
    
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    v3 = np.cross(v1, v2)
    v3 = v3 / np.linalg.norm(v3)
    
    cb = ca + 1.54 * v3  
    return np.round(cb, 3)

class coarse_graph_maker: 

    def __init__(self, filepath): 
        self.filepath = filepath 
        self.atom_type_map = {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'sN': 5, 'sC': 6, 'sO':7, 'S': 8, 'H': 9}
        self.aa_type_map = {
                        'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
                        'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
                        'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
                        'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
                    }
        self.res_type = torch.tensor([
            [0.33847796, 0.47396984, 0.36977478, 0.44351631],
            [0.32012752, 0.3754733,  1.00000000, 0.31650816],
            [0.33962618, 0.80456384, 0.28580494, 0.50466953],
            [0.38873317, 0.00000000, 0.47634157, 0.14178194],
            [1.00000000, 0.52110112, 0.00000000, 0.36948328],
            [0.00000000, 1.00000000, 0.38107325, 0.04312349],
            [0.10389838, 0.35895352, 0.09526831, 0.52336085],
            [0.36987242, 0.44515960, 0.38107325, 0.18816054],
            [0.01946829, 0.34503346, 0.19053663, 0.86588800],
            [0.35695416, 0.47315332, 0.33343910, 0.57953808],
            [0.37763855, 0.44370478, 0.52397572, 0.00000000],
            [0.41166668, 0.53687428, 0.42870741, 0.43844724],
            [0.21333724, 0.46011696, 0.23817078, 0.64365926],
            [0.39894287, 0.47758957, 0.28580494, 0.17976419],
            [0.37854265, 0.44960977, 0.47634157, 0.80904724],
            [0.27564020, 0.29129004, 0.38107325, 0.64645908],
            [0.43836698, 0.61005359, 0.57160988, 1.00000000],
            [0.56315348, 0.69166576, 0.54779280, 0.55852769],
            [0.30749454, 0.36081073, 0.33343910, 0.54861740],
            [0.16761799, 0.36027340, 0.09526831, 0.06977401]
        ])
        self.h_bond_donors = {
            'ARG': ['HE', '1HH1', '1HH2', '2HH1', '2HH2'],
            'LYS': ['1HZ', '2HZ', '3HZ'],
            'ASN': ['1HD2', '2HD2'],
            'GLN': ['1HE2', '2HE2'],
            'HIS': ['HD1', 'HE2'],
            'SER': ['HG'],
            'THR': ['HG1'],
            'TYR': ['HH'],
            'TRP': ['HE1'],
        }

        self.xyz, self.atmtp, self.seqsep, self.res, self.pepidx, self.peplen = self.parse_pdb()

    def parse_pdb(self):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('structure', self.filepath)

        xyz, atmtp, seqsep, res, pepidx = [],[],[],[], []
        current_seq = None
        current_chain = None
        seq_idx = 0

        idx = 0  # global atom index

        for model in structure:
            for chain in model:
                for residue in chain:
                    resseq = residue.get_id()[1]

                    atom_coords = {}
                    has_cb = False

                    for atom in residue:
                        resname = residue.get_resname()
                        if resname not in self.aa_type_map: continue
                        atom_name = atom.get_name().strip()
                        if atom_name == 'OXT': continue

                        if 'S' in atom_name:
                            atom_key = 'S'
                        elif atom_name in ['N', 'CA', 'C', 'O', 'CB', 'H']:
                            atom_key = atom_name
                        elif 'N' in atom_name: 
                            atom_key = 'sN'
                        elif 'C' in atom_name:
                            atom_key = 'sC'
                        elif 'O' in atom_name:
                            atom_key = 'sO'
                        elif resname in self.h_bond_donors.keys() and atom_name in self.h_bond_donors[resname]:
                            atom_key = 'H'
                        else:
                            continue
                        
                        if atom_key not in self.atom_type_map:
                            continue

                        atom_coords[atom_key] = atom.coord

                        if atom_key == 'CB':
                            has_cb = True

                        # 바로 저장
                        xyz.append(list(atom.coord))
                        atmtp.append(self.atom_type_map[atom_key])
                        res_idx = self.aa_type_map[resname]
                        res.append(res_idx)

                        if current_seq is None:
                            current_seq = resseq
                            current_chain = chain.id
                            seq_idx = 1
                        else:
                            if resseq != current_seq:
                                seq_idx += 1
                                current_seq = resseq
                            if chain.id != current_chain:
                                seq_idx += 200
                                current_chain = chain.id

                        seqsep.append(seq_idx)

                        if chain.id == 'X':
                            pepidx.append(idx)

                        idx += 1

                    # 만약 CB가 없으면 virtual CB 추가
                    if not has_cb and all(key in atom_coords for key in ['N', 'CA', 'C']):
                        virtual_cb = calculate_virtual_cb(atom_coords['N'], atom_coords['CA'], atom_coords['C'])
                        
                        xyz.append(list(virtual_cb))
                        atmtp.append(self.atom_type_map['CB'])  # CB index
                        seqsep.append(seq_idx)
                        res.append(res_idx)
                        
                        if chain.id == 'X':
                            pepidx.append(idx)
                        
                        idx += 1
        
        peplen = len(set(np.array(seqsep)[pepidx]))

        return torch.tensor(xyz, dtype=torch.float32), torch.tensor(atmtp, dtype=torch.long), torch.tensor(seqsep, dtype=torch.long), torch.tensor(res, dtype=torch.long), pepidx, peplen

    def make_graph(self, d_cut): 

        pep_mask = torch.zeros(len(self.atmtp), dtype=torch.bool)
        pep_mask[self.pepidx] = True

        binder_cond = pep_mask & ((self.atmtp == 1) | (self.atmtp == 4))
        non_binder_cond = (~pep_mask) & (self.atmtp == 1)
        final_mask = binder_cond | non_binder_cond

        xyz = self.xyz[final_mask]
        atmtp = self.atmtp[final_mask]
        seqsep = self.seqsep[final_mask]
        res = self.res[final_mask]

        survivors = final_mask.nonzero(as_tuple=False).squeeze()

        new_pepidx_mask = torch.isin(survivors, torch.tensor(self.pepidx))
        pepidx = new_pepidx_mask.nonzero(as_tuple=False).squeeze()

        pep_xyz = xyz[pepidx]
        com = pep_xyz.mean(dim=0, keepdim=True)
        xyz = xyz - com 

        dist = torch.norm(xyz, dim=1)
        dist_mask = dist <= d_cut 

        dist_mask[pepidx] = True 
        
        xyz = xyz[dist_mask]
        atmtp = atmtp[dist_mask]
        atmtp = (atmtp == 4).long()
        seqsep = seqsep[dist_mask]
        res = res[dist_mask]
        self.original_coords = xyz.clone()

        survivors = dist_mask.nonzero(as_tuple=False).squeeze()
        pepidx = torch.isin(survivors, pepidx).nonzero(as_tuple=False).squeeze()

        """ Nodes """
        is_pep = torch.zeros(xyz.size(0),1)
        is_pep[pepidx] = 1.0

        atom_type = torch.eye(2)[atmtp]
        res_type = self.res_type[res]
        res_type[pepidx,:] = torch.tensor(-1.0)
        node_attr = torch.cat([is_pep, res_type, atom_type], dim=1)
        nodes = torch.arange(len(node_attr))
        
        """ edges """
        edge_set = set()
        for p in pepidx.tolist(): 
            for i in range(xyz.size(0)): 
                if i != p:
                    edge_set.add((p,i))
                    edge_set.add((i,p))
        edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()

        seqsep_diff = torch.tanh(0.01*seqsep[:,None] - seqsep[None,:])
        u,v = edge_index
        edge_attr = seqsep_diff[u,v].unsqueeze(1)
        
        connection = torch.zeros(seqsep.shape[0], seqsep.shape[0], 2)
        atmtp = np.array(atmtp)
        pepCAidx = np.where(atmtp[pepidx]==0)[0]
        pepCBidx = np.where(atmtp[pepidx]==1)[0]
        pepCAidx = np.array(pepidx)[pepCAidx]
        pepCBidx = np.array(pepidx)[pepCBidx]
        pepCBCAidx = pepCBidx - 1
        peplength = len(pepCAidx)
        connection[pepCAidx[:-1],pepCAidx[1:], :] = torch.tile(torch.tensor([1,0], dtype=torch.float), (peplength-1, 1))
        connection[pepCAidx[1:],pepCAidx[:-1], :] = torch.tile(torch.tensor([1,0], dtype=torch.float), (peplength-1, 1))
        connection[pepCBCAidx, pepCBidx, :] = torch.tile(torch.tensor([0,1], dtype=torch.float), (peplength, 1))
        connection[pepCBidx, pepCBCAidx, :] = torch.tile(torch.tensor([0,1], dtype=torch.float), (peplength, 1))
        connection = connection[u,v]
        
        edge_attr = torch.cat([edge_attr,connection], dim=1)

        """ Graph """
        G = Data( 
            nodes = nodes,
            num_nodes = nodes.size(0),
            node_attr = node_attr,
            node_xyz = xyz, 
            edge_index = edge_index, 
            edge_attr = edge_attr, 
            pepidx = pepidx,
            seqidx = seqsep
        )

        return G, com
    

    def make_graph_seq(self, d_cut, resnum): 
        
        resnum = self.seqsep[self.pepidx].unique()[resnum].item()
        resnum_mask = (self.seqsep == resnum)
        
        pep_mask = torch.zeros(len(self.atmtp), dtype=torch.bool)
        pep_mask[self.pepidx] = True

        binder_cond = resnum_mask & ((self.atmtp == 1) | (self.atmtp == 4))
        non_binder_cond = (~pep_mask) & (self.atmtp == 1)
        final_mask = binder_cond | non_binder_cond

        xyz = self.xyz[final_mask]
        atmtp = self.atmtp[final_mask]
        seqsep = self.seqsep[final_mask]
        res = self.res[final_mask]
        survivors = final_mask.nonzero(as_tuple=False).squeeze()

        new_pepidx_mask = torch.isin(survivors, torch.tensor(self.pepidx))
        pepidx = new_pepidx_mask.nonzero(as_tuple=False).squeeze()

        com = xyz[pepidx[1]]
        xyz = xyz - com 

        dist = torch.norm(xyz, dim=1)
        dist_mask = dist <= d_cut 

        dist_mask[pepidx] = True 
        
        xyz = xyz[dist_mask]
        atmtp = atmtp[dist_mask]
        atmtp = (atmtp == 4).long()
        seqsep = seqsep[dist_mask]
        res = res[dist_mask]

        survivors = dist_mask.nonzero(as_tuple=False).squeeze()
        pepidx = torch.isin(survivors, pepidx).nonzero(as_tuple=False).squeeze()

        """ Nodes """
        is_pep = torch.zeros(xyz.size(0),1)
        is_pep[pepidx] = 1.0

        res_type = torch.eye(20)[res]
        atom_type = torch.eye(2)[atmtp]
        node_attr = torch.cat([is_pep, res_type, atom_type], dim=1)
        nodes = torch.arange(len(node_attr))
        
        """ edges """
        edge_set = set()
        for p in pepidx.tolist(): 
            for i in range(xyz.size(0)): 
                if i != p: 
                    edge_set.add((p,i))
                    edge_set.add((i,p))
        edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()
        
        seqsep_diff = torch.tanh(0.01*seqsep[:,None] - seqsep[None,:])
        dist_diff = torch.cdist(xyz,xyz,p=2)
        u,v = edge_index
        edge_attr = torch.cat([seqsep_diff[u,v].unsqueeze(1), dist_diff[u,v].unsqueeze(1)], dim=1)

        # edge_attr = torch.cat([edge_attr], dim=1)

        """ Graph """
        G = Data( 
            nodes = nodes,
            num_nodes = nodes.size(0),
            node_attr = node_attr,
            node_xyz = xyz, 
            edge_index = edge_index, 
            edge_attr = edge_attr, 
            pepidx = pepidx,
            seqidx = seqsep
        )

        return G, com


