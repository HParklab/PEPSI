import numpy as np
import torch
from typing import Tuple, List
from torch import Tensor
from torch_geometric.data import Data
from Bio.PDB import PDBParser



class coarse_graph_maker: 

    def __init__(self, filepath:str, chain_ID:str='X') -> None: 
        """
        Initialize the class with input file and predefined atom/amino acid mappings.

        Args:
            filepath (str): Path to the input PDB file.

        Attributes:
            filepath (str): File path passed to the class.
            atom_type_map (dict): Mapping from atom names to indices.
            aa_type_map (dict): Mapping from 3-letter amino acid codes to indices.
            res_type (Tensor): Residue-level physicochemical property matrix (20 residues × 4 features).
            h_bond_donors (dict): Residue-wise dictionary of atom names that serve as hydrogen bond donors.

        Returns:
            None
        """
        self.chain_ID = chain_ID
        self.filepath = filepath # Path to the input structure file
        self.atom_type_map = {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'sN': 5, 'sC': 6, 'sO':7, 'S': 8, 'H': 9}
        self.aa_type_map = {
                        'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
                        'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
                        'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
                        'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
                    }
        
        # Amino Acid type features : One-Hot => PCA
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

        # Atom names for each residue type that can act as hydrogen bond donors
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

    def calculate_virtual_cb(self, n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
        """
        Calculate a virtual CB (beta carbon) position from N, CA, and C atom coordinates.

        Args:
            n (np.ndarray): Coordinate of the backbone nitrogen (N) atom. Shape: (3,)
            ca (np.ndarray): Coordinate of the alpha carbon (CA) atom. Shape: (3,)
            c (np.ndarray): Coordinate of the carbon (C) atom. Shape: (3,)

        Returns:
            np.ndarray: Estimated CB coordinate, rounded to 3 decimal places. Shape: (3,)
        """
        v1 = n - ca 
        v2 = c - ca  
        
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)

        v3 = np.cross(v1, v2)
        v3 = v3 / np.linalg.norm(v3)
        
        cb = ca + 1.54 * v3  
        return np.round(cb, 3)

    def parse_pdb(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, List, int]:
        """
        Parse a PDB file and extract atom-level and residue-level information.

        Args:
            None (uses self.filepath internally)

        Returns:
            - xyz (Tensor): Atom coordinates (N_atoms, 3)
            - atmtp (Tensor): Atom type indices (N_atoms,)
            - seqsep (Tensor): Sequence separation (residue index along chain)
            - res (Tensor): Residue type indices (N_atoms,)
            - pepidx (List[int]): Indices of atoms that belong to chain 'X' (peptide)
        """
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('structure', self.filepath)

        xyz, atmtp, seqsep, res, pepidx = [],[],[],[], []
        current_seq = None # Track residue number
        current_chain = None # Track chain ID
        seq_idx = 0 # Sequence index (used for seqsep)
        idx = 0  # global atom index

        for model in structure:
            for chain in model:
                for residue in chain:
                    resseq = residue.get_id()[1] # Residue number

                    atom_coords = {} # Store N/CA/C atom coords for virtual CB
                    has_cb = False # Flag to check if CB exists

                    for atom in residue:
                        resname = residue.get_resname() # e.g., 'ALA', 'ARG'
                        if resname not in self.aa_type_map: continue # Skip non-standard residues

                        atom_name = atom.get_name().strip() # e.g., 'CA', 'CB'
                        if atom_name == 'OXT': continue # Skip OXT terminal atom
                        # Atom name => atom_key mapping
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
                            has_cb = True # Mark if CB exists

                        xyz.append(list(atom.coord))
                        atmtp.append(self.atom_type_map[atom_key])
                        res_idx = self.aa_type_map[resname]
                        res.append(res_idx)

                        # Track sequence position across chains and residues
                        if current_seq is None:
                            current_seq = resseq
                            current_chain = chain.id
                            seq_idx = 1
                        else:
                            if resseq != current_seq:
                                seq_idx += 1
                                current_seq = resseq
                            if chain.id != current_chain:
                                seq_idx += 200 # Chain offset if ChainID is changed
                                current_chain = chain.id
                        seqsep.append(seq_idx)
                        
                        # Mark Ppetide Atoms
                        if chain.id == self.chain_ID:
                            pepidx.append(idx)

                        idx += 1 # Increase Global Atom index

                    # Add virtual CB if missing and N/CA/C are presentåå
                    if not has_cb and all(key in atom_coords for key in ['N', 'CA', 'C']):
                        virtual_cb = self.calculate_virtual_cb(atom_coords['N'], atom_coords['CA'], atom_coords['C'])
                        # Append virtual CB
                        xyz.append(list(virtual_cb))
                        atmtp.append(self.atom_type_map['CB'])  # CB index
                        seqsep.append(seq_idx)
                        res.append(res_idx)
                        if chain.id == self.chain_ID:
                            pepidx.append(idx)
                        idx += 1
        
        peplen = len(set(np.array(seqsep)[pepidx]))
        self.peplen = peplen

        return torch.tensor(xyz, dtype=torch.float32), torch.tensor(atmtp, dtype=torch.long), torch.tensor(seqsep, dtype=torch.long), torch.tensor(res, dtype=torch.long), pepidx

    def make_graph(self, d_cut:float) -> Tuple[Data, Tensor]: 
        """
        Build a graph representation of the protein-peptide complex.

        Args:
            d_cut (float): Distance cutoff for filtering atoms based on their distance
                        to the peptide center-of-mass.

        Returns:
            Tuple[
                Data: PyTorch Geometric Data object representing the graph,
                Tensor: Center-of-mass coordinates of the peptide (1, 3)
            ]
        """
        xyz, atmtp, seqsep, res, pepidx = self.parse_pdb()

        """ Filtering """
        pep_mask = torch.zeros(len(atmtp), dtype=torch.bool)
        pep_mask[pepidx] = True

        binder_cond = pep_mask & ((atmtp == 1) | (atmtp == 4)) # Peptide atoms with CA or CB
        non_binder_cond = (~pep_mask) & (atmtp == 1)           # Non-peptide atoms with CA
        final_mask = binder_cond | non_binder_cond             # Atoms to keep for graph

        # filter only CA if receptor, CA+CB if binder
        xyz = xyz[final_mask]
        atmtp = atmtp[final_mask]
        seqsep = seqsep[final_mask]
        res = res[final_mask]

        survivors = final_mask.nonzero(as_tuple=False).squeeze() # Indices of selected atoms
        new_pepidx_mask = torch.isin(survivors, torch.tensor(pepidx)) # Update peptide indices
        pepidx = new_pepidx_mask.nonzero(as_tuple=False).squeeze()

        pep_xyz = xyz[pepidx]
        com = pep_xyz.mean(dim=0, keepdim=True)
        xyz = xyz - com # Center structure around total peptide

        dist = torch.norm(xyz, dim=1)
        dist_mask = dist <= d_cut # Mask atoms within cutoff from (0,0,0)

        dist_mask[pepidx] = True # Always include peptide atoms
        
        # filter within distance cutoff
        xyz = xyz[dist_mask]
        atmtp = atmtp[dist_mask]
        atmtp = (atmtp == 4).long()
        seqsep = seqsep[dist_mask]
        res = res[dist_mask]

        survivors = dist_mask.nonzero(as_tuple=False).squeeze() # Indices of filtered atoms
        pepidx = torch.isin(survivors, pepidx).nonzero(as_tuple=False).squeeze() # Update peptide indices

        """ Nodes """
        is_pep = torch.zeros(xyz.size(0),1)      # Binary indicator for peptide atoms
        is_pep[pepidx] = 1.0

        atom_type = torch.eye(2)[atmtp]          # One-hot for CA(0) or CB(1)
        res_type = self.res_type[res]            # Amino Acid type features
        res_type[pepidx,:] = torch.tensor(-1.0)  # Mask peptide residues
        node_attr = torch.cat([is_pep, res_type, atom_type], dim=1)
        nodes = torch.arange(len(node_attr))     # Node indices
        
        """ edges """
        # Fully-Connected Edge connection
        edge_set = set() # Delete duplication
        for p in pepidx.tolist(): 
            for i in range(xyz.size(0)): 
                if i != p:
                    edge_set.add((p,i)) # Biderectional Edge
                    edge_set.add((i,p))
        edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous() # Shape: (2,E)

        seqsep_diff = torch.tanh(0.01*seqsep[:,None] - seqsep[None,:]) # Sequence idx difference feature
        u,v = edge_index
        edge_attr = seqsep_diff[u,v].unsqueeze(1)
        
        connection = torch.zeros(seqsep.shape[0], seqsep.shape[0], 2) # Chemical Bond Binary feature
        atmtp = np.array(atmtp)
        pepCAidx = np.array(pepidx)[np.where(atmtp[pepidx]==0)[0]]
        pepCBidx = np.array(pepidx)[np.where(atmtp[pepidx]==1)[0]]
        pepCBCAidx = pepCBidx - 1 # CAidx which comes right before CBidx
        peplength = len(pepCAidx)

        connection[pepCAidx[:-1],pepCAidx[1:], :] = torch.tile(torch.tensor([1,0], dtype=torch.float), (peplength-1, 1)) # CA-CA connection
        connection[pepCAidx[1:],pepCAidx[:-1], :] = torch.tile(torch.tensor([1,0], dtype=torch.float), (peplength-1, 1))
        connection[pepCBCAidx, pepCBidx, :] = torch.tile(torch.tensor([0,1], dtype=torch.float), (peplength, 1)) # CA-CB connection
        connection[pepCBidx, pepCBCAidx, :] = torch.tile(torch.tensor([0,1], dtype=torch.float), (peplength, 1))
        connection = connection[u,v]
        
        edge_attr = torch.cat([edge_attr,connection], dim=1) # Distance feature will be added during training & sampling

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


