import torch, sys, os
import numpy as np
from tqdm import tqdm
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_dir)
from util.sampling_utils import *
from util.arguments import set_arguments
from util.pdb_parsing import *
from util.utils import *


args = set_arguments()
oripath = '/home/jsi0613/projects/ddpm/data/refined_peplen_8-18_interaction50/'
samplepath = '/home/jsi0613/projects/ddpm_coarse/data/8-18_x0pdl_samples/'
samplelist = os.listdir(samplepath)


CACA_length_ori, CACA_length_x0, CACB_length_ori, CACB_length_x0, minCA_length_ori, minCA_length_x0, num_interaction_ori, num_interaction_x0 = [],[],[],[],[],[],[],[]
for pdb in tqdm(samplelist[:5]): 
    print(pdb)
    oripdb = oripath + pdb 
    ori_graph = coarse_graph_maker(oripdb)
    ori_G,_ = ori_graph.make_graph3(25.0)

    samplepdb = samplepath + pdb 
    x_t1 = pdb_to_all_coords_tensor(samplepdb)
    x_t1 = x_t1 - x_t1.mean(dim=0)

    ori = ori_G.node_xyz
    pepidx = ori_G.pepidx.to(device=ori.device)
    ori = ori - ori[pepidx].mean(dim=0)

    x0 = ori.clone()
    x0[pepidx] = x_t1.to(device=ori.device)

    CACAori, CACBori = get_coarse_length(ori[pepidx])
    CACAx0, CACBx0 = get_coarse_length(x0[pepidx])
    ori_interaction, minCA_ori = get_coarse_interaction(ori, pepidx)
    x0_interaction, minCA_x0 = get_coarse_interaction(x0, pepidx)

    CACAori_mean = CACAori.mean()
    CACAori_var = CACAori.var() 
    CACBori_mean = CACBori.mean()
    CACBori_var = CACBori.var() 

    CACAx0_mean = CACAx0.mean()
    CACAx0_var = CACAx0.var() 
    CACBx0_mean = CACBx0.mean()
    CACBx0_var = CACBx0.var()

    CACA_length_ori.append(CACAori_mean.item())
    CACA_length_x0.append(CACAx0_mean.item())
    
    CACB_length_ori.append(CACBori_mean.item()) 
    CACB_length_x0.append(CACBx0_mean.item())

    minCA_length_ori.append(minCA_ori.item())
    minCA_length_x0.append(minCA_x0.item())

    num_interaction_ori.append(ori_interaction)
    num_interaction_x0.append(x0_interaction)

print(CACA_length_ori, CACA_length_x0)
print(CACB_length_ori, CACB_length_x0)
print(minCA_length_ori, minCA_length_x0)
print(num_interaction_ori, num_interaction_x0)