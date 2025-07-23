import torch, sys, os, pickle
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx 
from torch_geometric.data import Data
from tqdm import tqdm
import matplotlib.pyplot as plt
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_dir)
from models.egnn import EGNN
from util.sampling_utils import *
from util.diffusion import Diffusion
from arguments import set_arguments
from util.pdb_parsing import *
import shutil
import yaml


args = set_arguments()

config_path = args.project_path + "configs/"
model_name = args.model_name 
with open(config_path + model_name + '.yaml', 'r') as f: 
    config = yaml.load(f, Loader=yaml.FullLoader)
model_params = config["model_params"]
model_params["device"] = args.device 

args = set_arguments()
path = '/home/jsi0613/projects/ddpm/data/refined_peplen_8-18_interaction50/'
# path = '/scratch/jsi0613/CG_data/pdbs/'

pdbnum = os.listdir(path)[-8]
print(pdbnum)
shutil.copy(path+pdbnum, args.sample_path+pdbnum)
print(args.model_name, args.model_path)
sampling = sampling_code(
    EGNN, 
    model_params, 
    args.model_name, 
    args.model_path, 
    args.device, 
    args.timestep, 
    args.t_dim, 
    path, 
    args.sample_path) 
x_t1 = sampling.sample_pdb(pdbnum)


CACAx0, CACBx0 = get_coarse_length(x_t1)
print(min(CACBx0), max(CACBx0))





