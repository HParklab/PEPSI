import sys
import os
import pickle
from tqdm import tqdm
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_dir)
from util.pdb_parsing import *
from util.sampling_utils import *
from util.arguments import set_arguments
from util.utils import *
from pathlib import Path


args = set_arguments()
pdbpath = Path('/scratch/jsi0613/CG_data/pdbs') 
pklpath = Path('/scratch/jsi0613/CG_data/pickles')

for pdb in tqdm(list(pdbpath.iterdir())):
    pdbID = pdb.stem 
    to_graph = coarse_graph_maker(pdb) 

    G, com = to_graph.make_graph3(30.0)
    with open(pklpath.joinpath(pdbID + ".pkl"), 'wb') as f:
        pickle.dump(G,f)



#GPU2
# Gp = []
# for pdb in tqdm(newlist):
#     to_graph = coarse_graph_maker(newpath + pdb)
#     peplen = to_graph.peplen

#     for resnum in range(1,peplen-1):
#         G, com = to_graph.make_graph_seq(12.0, resnum)
#         Gp.append(G)
# with open('/scratch/jsi0613/refined_peplen_8-25_interaction50/sequence/coarse_graph_maker_mini.pkl', 'wb') as f:
#     pickle.dump(Gp,f)
# print(len(Gp))

#GPU3
# Gp = []
# for pdb in tqdm(newlist):
#     to_graph = coarse_graph_maker(newpath + pdb)
#     peplen = to_graph.peplen

#     G, com = to_graph.make_graph3(30.0)
#     Gp.append(G)
    
# with open('/scratch/jsi0613/refined_peplen_8-25_interaction50/coords/coarse_graph_maker.pkl', 'wb') as f:
#     pickle.dump(Gp,f)
# print(len(Gp))



