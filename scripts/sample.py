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
from util.sampling_utils import *
from util.diffusion import Diffusion
from util.arguments import set_arguments
from util.pdb_parsing import *
from dataset import collate
from util.utils import *



args = set_arguments()
# model_params = {
#                 'in_node_nf': 11,
#                 'hidden_nf': 160,
#                 'out_node_nf': 11,
#                 'in_edge_nf': 4,
#                 'device': torch.device('cuda'),
#                 'n_layers': 8
#                 }
model_params = {
                'in_node_nf': 11,
                'hidden_nf': 120,
                'out_node_nf': 11,
                'in_edge_nf': 4,
                'device': torch.device('cuda'),
                'n_layers': 12
                }
print(model_params)
    

args = set_arguments()
path = '/home/jsi0613/projects/ddpm/data/refined_peplen_8-18_interaction50/'

pdbnum = os.listdir(path)[-3]
print(pdbnum)


sampling = sampling_code(args, model_params, path, pdbnum, 'make_graph3') 
print("original PDB binder length : ", sampling.to_graph.peplen)
x_t1 = sampling.sample_pdb()


CACAx0, CACBx0 = get_coarse_length(x_t1)
print(min(CACBx0), max(CACBx0))





