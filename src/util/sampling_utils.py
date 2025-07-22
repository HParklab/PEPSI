import torch
import numpy as np
from torch import Tensor
from typing import List, Dict, Tuple
import os
from util.training_utils import dist_edges
from util.pdb_parsing import *
from util.diffusion import Diffusion
from models.egnn import EGNN


def load_best_model(model_class:EGNN, model_params:Dict, args, message=None):

    model_name = args.model_name
    model_path = args.model_path
    device = args.device
    model = model_class( **model_params )
    
    if os.path.exists(model_path+model_name+'/model.pkl'):
        checkpoint = torch.load(model_path+model_name+'/best.pkl',map_location=device)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        if message == None:
            print('model: ', model_name + ' is loaded')
            print(f'Best Epoch is {epoch}')

        return model
        
    else:
        print(f"there's no {model_name}")
        
        return

class sampling_code: 

    def __init__(self, args, model_params, pdb_path):

        self.args = args
        self.model_params = model_params
        self.pdb_path = pdb_path

    def preset(self, pdbnum):

        if type(pdbnum) == int:
                pdbs = os.listdir(self.pdb_path)
                pdbs = pdbs[int(len(pdbs)*0.8):]
                pdb = self.pdb_path + pdbs[pdbnum]
        elif type(pdbnum) == str: 
            pdb = self.pdb_path + pdbnum

        to_graph = coarse_graph_maker(pdb)
        G, com = to_graph.make_graph(30.0)
        self.to_graph = to_graph
        print(f'Peptide Length : {to_graph.peplen}')

        xyz = G.node_xyz.clone()
        ep = torch.randn_like(xyz[G.pepidx])
        ep = ep - torch.mean(ep, dim=0, keepdim=True)
        xyz[G.pepidx] = ep 
        G.node_xyz = xyz 

        G = dist_edges(G).to(self.args.device)

        return G,com

    def sample_pdb(self, pdbnum, traj=True, sample_pdb=None):

        G,com = self.preset(pdbnum)
        model = load_best_model(EGNN, self.model_params, self.args)

        ddpm = Diffusion(self.args.device, self.args.timestep, G)
        
        for t in reversed(range(self.args.timestep)): 

            base = ddpm.time_embedding(self.args.t_dim, t)
            G.node_attr = torch.cat([G.node_attr, base], dim=1)

            x_tu = G.node_xyz.clone()
            x_td = G.node_xyz[G.pepidx].clone()

            with torch.no_grad(): 
                _,x0 = model(G.node_attr, G.node_xyz, G.edge_index, G.edge_attr)
            x0_target = x0[G.pepidx]
            x0_target = x0_target - torch.mean(x0_target, dim=0, keepdim=True)

            x_t1 = ddpm.reverse_process(x_td, x0_target, t)

            G.node_xyz = x_tu 
            G.node_xyz[G.pepidx] = x_t1 

            G.node_attr = G.node_attr[:, :G.node_attr.shape[-1]-self.args.t_dim]
            G.edge_attr = G.edge_attr[:,1:]
            G = dist_edges(G)

            atom_types = torch.argmax(G.node_attr[G.pepidx, 5:7], dim=1)
            
            # if traj:
            #     self.sample2pdb_traj(self.args.sample_path, "sample_traj.pdb", atom_types, G.seqidx[G.pepidx], x_t1, com, G.pepidx, self.args.timestep-t)
            if t == 0:
                if sample_pdb == None:
                    self.sample2pdb(self.args.sample_path, 'sample.pdb', atom_types, G.seqidx[G.pepidx], x_t1, com, G.pepidx)
                else: 
                    self.sample2pdb(self.args.sample_path, sample_pdb, atom_types, G.seqidx[G.pepidx], x_t1, com, G.pepidx)

        return x_t1
    
                
    def sample2pdb(self, path, pdb_file, atom_types, seqidx, xyz_coord, com, pepidx, connect=True):

        pdb_file = path + pdb_file

        with open(pdb_file, 'w') as f:
            for i in range(len(pepidx)):
                chainID = 'X' 

                atom_idx = atom_types[i].item()
                seq_idx = int(seqidx[i].item()) + 1
                xyz = (xyz_coord[i] + com.to(device=xyz_coord.device)).squeeze(0)
                x, y, z = xyz.tolist()

                if atom_idx == 0:
                    f.write(f"ATOM  {i+1:5d}  CA  UNK {chainID}{int(seq_idx):4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
                elif atom_idx == 1:
                    f.write(f"ATOM  {i+1:5d}  CB  UNK {chainID}{int(seq_idx):4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
            
            if connect:
                CA_idx = list(range(1, i, 2))
                for i in range(len(CA_idx)):
                    a1 = CA_idx[i]
                    a2 = CA_idx[i] + 2
                    f.write(f"CONECT{a1:5d}{a2:5d}\n")

    def sample2pdb_traj(self, path, pdb_file, atom_types, seqidx, xyz_coord, com, pepidx, model_num):

        pdb_file = path + pdb_file

        with open(pdb_file, 'a') as f:
            f.write(f"MODEL     {model_num+1:>4}\n")
            for i in range(len(pepidx)):
                chainID = 'X' 

                atom_idx = atom_types[i].item()
                seq_idx = int(seqidx[i].item()) + 1
                xyz = (xyz_coord[i] + com.to(device=xyz_coord.device)).squeeze(0)
                x, y, z = xyz.tolist()

                if atom_idx == 0:
                    f.write(f"ATOM  {i+1:5d}  H   UNK {chainID}{int(seq_idx):4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           H\n")
                elif atom_idx == 1:
                    f.write(f"ATOM  {i+1:5d}  H   UNK {chainID}{int(seq_idx):4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           H\n")
            f.write("ENDMDL\n")
            

def get_coarse_length(x0):

    CAx0 = x0[::2]
    CBx0 = x0[1::2]

    CACAx0 = torch.sqrt(torch.sum((CAx0[1:] - CAx0[:-1])**2, dim=1))
    CACBx0 = torch.sqrt(torch.sum((CBx0 - CAx0)**2, dim=1))

    return CACAx0, CACBx0

def get_coarse_interaction(x, idx): 

    idx_mask = torch.zeros(len(x), dtype=torch.bool)
    idx_mask[idx] = True

    rec = x[~idx_mask]
    bin = x[idx_mask]

    dist_map = torch.cdist(rec, bin, p=2)
    interacting_mask = (dist_map < 6.0).any(dim=1)

    num_interaction = interacting_mask.sum().item()

    bin = bin[::2]
    dist_map = torch.cdist(bin, rec, p=2)
    minCA_dist = torch.mean(torch.min(dist_map, dim=1)[0])

    return num_interaction, minCA_dist

def pdb_to_all_coords_tensor(pdb_path):

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)

    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coords.append(atom.get_coord())

    return torch.tensor(coords, dtype=torch.float32)