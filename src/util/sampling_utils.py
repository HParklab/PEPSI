import torch
import os
import torch.nn as nn
from util.training_utils import dist_edges
from util.pdb_parsing import coarse_graph_maker
from util.diffusion import Diffusion
from models.egnn import EGNN
from torch_geometric.data import Data
from Bio.PDB import PDBParser
from torch import Tensor
from typing import Dict, Tuple, Type

def load_best_model(model_class:Type[nn.Module], model_params:Dict, model_name:str, model_path:str, device:str, message:str=None) -> nn.Module:
    """
    Load a model instance and its best checkpoint if available.

    Args:
        model_class (Type[nn.Module]): The class of the model to instantiate.
        model_params (Dict): Keyword arguments for model initialization.
        model_name (str): Name of the model directory (e.g., 'EGNN_v1').
        model_path (str): Base path where the model directory is located.
        device (torch.device): Device on which to load the model (e.g., 'cuda' or 'cpu').
        message (str, optional): If None, print loading message; otherwise silent.

    Returns:
        nn.Module: The model with weights loaded if checkpoint exists; otherwise a freshly initialized model.
    """
    model = model_class( **model_params )
    
    if os.path.exists(model_path+model_name+'/model.pkl'):
        checkpoint = torch.load(model_path+model_name+'/best.pkl',map_location=device) # Load Best checkpoint
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

    def __init__(self, model_params:Dict, model_name:str, model_path:str, device:str, timestep:int, t_dim:int, pdb_path:str, sample_path:str) -> None:
        """
        Initialize the model wrapper with configuration parameters.

        Args:
            model_params (dict): Dictionary of model hyperparameters for initialization.
            model_name (str): Name of the model (used for saving/loading checkpoints).
            model_path (str): Path where the model checkpoints will be stored or loaded from.
            device (torch.device): Computation device ('cuda' or 'cpu').
            timestep (int): Total number of diffusion steps (T).
            t_dim (int): Dimensionality of time embedding or timestep input.
            pdb_path (str): Path to the input PDB file or directory.
            sample_path (str): Directory where generated samples will be saved.
        """
        self.model_params = model_params
        self.model_name = model_name 
        self.model_path = model_path 
        self.device = device
        
        self.timestep = timestep
        self.t_dim = t_dim

        self.pdb_path = pdb_path 
        self.sample_path = sample_path

    def preset(self, pdbnum:str|int) -> Tuple[Data, Tensor]:
        """
        Prepare a graph from a selected PDB file and initialize the peptide coordinates with noise.

        Args:
            pdbnum (str or int): If int, index into the last 20% of PDB files in self.pdb_path.
                                If str, use the exact PDB filename (relative to self.pdb_path).

        Returns:
            Tuple[Data, Tensor]: 
                - Data: PyTorch Geometric Data object representing the graph.
                - Tensor: The center of mass of the peptide (shape: [1, 3]).
        """
        if type(pdbnum) == int:
                pdbs = os.listdir(self.pdb_path)
                pdbs = pdbs[int(len(pdbs)*0.8):]
                pdb = self.pdb_path + pdbs[pdbnum]
        elif type(pdbnum) == str: 
            pdb = self.pdb_path + pdbnum

        to_graph = coarse_graph_maker(pdb)
        G, com = to_graph.make_graph(30.0)
        print(f'Peptide Length : {to_graph.peplen}')

        xyz = G.node_xyz.clone()
        ep = torch.randn_like(xyz[G.pepidx]) # Generate Gaussian noise => X_T
        ep = ep - torch.mean(ep, dim=0, keepdim=True)
        xyz[G.pepidx] = ep 
        G.node_xyz = xyz # Update Graph

        G = dist_edges(G).to(self.device) # Add distance edge feature

        return G, com

    def sample_pdb(self, pdbnum:str|int, traj:bool=False, sample_pdb:str=None) -> Tensor:
        """
        Generate a peptide structure via reverse diffusion and optionally save trajectory or final PDB.

        Args:
            pdbnum (str or int): PDB input identifier. If int, samples from dataset; if str, uses exact PDB name.
            traj (bool, optional): Whether to save the full trajectory as a multi-frame PDB. Default is False.
            sample_pdb (str, optional): Name for saving the final sample PDB. If None, uses 'sample.pdb'.

        Returns:
            Tensor: Final denoised peptide coordinates of shape (N_peptide_atoms, 3)
        """
        G, com = self.preset(pdbnum)
        model = load_best_model(EGNN, self.model_params, self.model_name, self.model_path, self.device)
        ddpm = Diffusion(self.device, self.timestep, G)
        
        for t in reversed(range(self.timestep)): # Reverse-Time sampling loop

            base = ddpm.time_embedding(self.t_dim, t) # Time embedding for current timestep
            G.node_attr = torch.cat([G.node_attr, base], dim=1)

            x_tu = G.node_xyz.clone()           # Save all coords
            x_td = G.node_xyz[G.pepidx].clone() # Save peptide coords

            with torch.no_grad(): 
                _,x0 = model(G.node_attr, G.node_xyz, G.edge_index, G.edge_attr) # Predict x0 
            x0_target = x0[G.pepidx] # Only use predicted x0 of peptide nodes
            x0_target = x0_target - torch.mean(x0_target, dim=0, keepdim=True)

            x_t1 = ddpm.reverse_process(x_td, x0_target, t) # One step reverse diffusion

            G.node_xyz = x_tu 
            G.node_xyz[G.pepidx] = x_t1 # Update only peptide part

            G.node_attr = G.node_attr[:, :G.node_attr.shape[-1]-self.t_dim] # Remove time embedding
            G.edge_attr = G.edge_attr[:,1:] # Remove old distance edge feature
            G = dist_edges(G) # Add distance edge feature of updated coords

            atom_types = torch.argmax(G.node_attr[G.pepidx, 5:7], dim=1)
            
            if traj:
                self.sample2pdb_traj(self.sample_path, "sample_traj.pdb", atom_types, G.seqidx[G.pepidx], x_t1, com, G.pepidx, self.timestep-t)
            if t == 0:
                if sample_pdb == None:
                    self.sample2pdb(self.sample_path, 'sample.pdb', atom_types, G.seqidx[G.pepidx], x_t1, com, G.pepidx)
                else: 
                    self.sample2pdb(self.sample_path, sample_pdb, atom_types, G.seqidx[G.pepidx], x_t1, com, G.pepidx)

        return x_t1
    
                
    def sample2pdb(self, path:str, pdb_file:str, atom_types:Tensor, seqidx:Tensor, xyz_coord:Tensor, com:Tensor, pepidx:Tensor, connect:bool=True) -> None:
        """
        Save generated peptide structure to a PDB file.

        Args:
            path (str): Directory path where the PDB will be saved.
            pdb_file (str): File name for the output PDB file.
            atom_types (Tensor): Tensor of shape (N,) indicating atom type (0 = CA, 1 = CB).
            seqidx (Tensor): Sequence indices for each peptide atom.
            xyz_coord (Tensor): Predicted coordinates of shape (N, 3).
            com (Tensor): Center-of-mass vector used to un-center the coordinates.
            pepidx (Tensor): Indices of peptide atoms in the original graph.
            connect (bool, optional): Whether to add CONECT records for CA backbone. Default is True.
        """
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

    def sample2pdb_traj(self, path:str, pdb_file:str, atom_types:Tensor, seqidx:Tensor, xyz_coord:Tensor, com:Tensor, pepidx:Tensor, model_num:int) -> None:
        """
        Append a sampled peptide structure to a multi-model PDB trajectory file.

        Args:
            path (str): Directory where the PDB file will be saved.
            pdb_file (str): File name of the trajectory PDB file.
            atom_types (Tensor): Tensor indicating atom types (0 = CA, 1 = CB).
            seqidx (Tensor): Tensor of sequence indices for each atom.
            xyz_coord (Tensor): Predicted coordinates of atoms (N, 3).
            com (Tensor): Center-of-mass to shift coordinates back to original position.
            pepidx (Tensor): Indices of peptide atoms.
            model_num (int): Index of the model (frame) in the trajectory.
        """
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
            

def get_coarse_length(x0:Tensor) -> Tuple[Tensor, Tensor]:
    """
    Compute backbone distances from a coarse-grained structure tensor.

    Args:
        x0 (Tensor): Tensor of coordinates shaped (2N, 3) where even indices are CA and odd are CB atoms.

    Returns:
        Tuple[Tensor, Tensor]:
            - CACAx0: Distances between consecutive CA atoms.
            - CACBx0: Distances between CA and CB within each residue.
    """
    CAx0 = x0[::2]
    CBx0 = x0[1::2]

    CACAx0 = torch.sqrt(torch.sum((CAx0[1:] - CAx0[:-1])**2, dim=1))
    CACBx0 = torch.sqrt(torch.sum((CBx0 - CAx0)**2, dim=1))

    return CACAx0, CACBx0

def get_coarse_interaction(x:Tensor, idx:Tensor) -> Tuple[Tensor, Tensor]: 
    """
    Compute number of receptor atoms near binder, and average closest CA distance.

    Args:
        x (Tensor): All atom coordinates, shape (N, 3).
        idx (Tensor): Index tensor for binder atoms in x.

    Returns:
        Tuple[Tensor, Tensor]:
            - num_interaction (int): Number of receptor atoms within 6 Å of any binder atom.
            - minCA_dist (float): Average minimum distance from binder CA atoms to receptor atoms.
    """
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

def pdb_to_all_coords_tensor(pdb_path:str) -> Tensor:
    """
    Extract all atom coordinates from a PDB file.

    Args:
        pdb_path (str): Path to the PDB file.

    Returns:
        Tensor: Tensor of all atom coordinates with shape (N_atoms, 3).
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)

    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coords.append(atom.get_coord())

    return torch.tensor(coords, dtype=torch.float32)