import torch
from torch_geometric.data import Batch
from util.pdb_parsing import *
import pickle
from tqdm import tqdm
import time
from torch import Tensor
from typing import List, Dict, Tuple

class DataSet(torch.utils.data.Dataset):

    def __init__(self, data_path) -> None:
        self.fs = []
        start_time = time.time()
        for pkl in list(data_path.iterdir())[:30]: 
            with open(pkl, 'rb') as f: 
                data = pickle.load(f)
                self.fs.append(data)
        end_time = time.time() 
        print("Loading Time : ", int(end_time - start_time))
        
    def __len__(self):
        return len(self.fs)
        
    def __getitem__(self, index:int):
        
        ipdb = index%len(self.fs)
        G = self.fs[ipdb]
        
        return G


def collate(samples): 

    bG = Batch.from_data_list(samples) 

    pepidx_list = []
    pepidx2_list = []
    offset = 0
    for i,data in enumerate(samples): 
        pepidx_list.append(data.pepidx + offset)
        pepidx2_list.append(data.pepidx.tolist())
        offset += data.num_nodes 
    bG.pepidx = torch.cat(pepidx_list)
    bG.pepidx2 = pepidx2_list

    return bG