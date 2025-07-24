import torch, os, sys, shutil
import numpy as np
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_dir)
from util.sampling_utils import sampling_code, get_coarse_length
from models.egnn import EGNN

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run sampling with pretrained model")
    parser.add_argument('--pdb_path', type=str, required=True, help='Directory with input PDB files')
    parser.add_argument('--pdbnum', type=str, required=True, help='PDB file name to sample (e.g., 1abc.pdb)')
    parser.add_argument('--chainID', type=str, required=True)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu" 
    t_dim = 4 
    timestep = 300 
    sample_path = './samples/'
    model_path = './params/'
    model_name = 'Coarse-Grained'


    # Load YAML model config
    model_params = {
        'attention': True,
        'hidden_nf': 160,
        'in_edge_nf': 4,
        'in_node_nf': 11,
        'n_layers': 8,
        'out_node_nf': 11,
        'device': device
    }

    # Copy input PDB to sample_path for record-keeping
    os.makedirs(sample_path, exist_ok=True)
    shutil.copy(os.path.join(args.pdb_path, args.pdbnum), os.path.join(sample_path, args.pdbnum))

    # Initialize sampler
    sampler = sampling_code(
        EGNN, model_params,
        model_name=model_name,
        model_path=model_path,
        device=device,
        timestep=timestep,
        t_dim=t_dim,
        pdb_path=args.pdb_path,
        sample_path=sample_path
    )

    # Run sampling
    x_t1 = sampler.sample_pdb(args.pdbnum, args.chainID)

    # Optionally: print coarse distances
    CACAx0, CACBx0 = get_coarse_length(x_t1)
    print(f"Min CA–CB distance: {min(CACBx0):.2f}, Max: {max(CACBx0):.2f}")