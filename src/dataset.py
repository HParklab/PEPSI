import torch
from torch_geometric.data import Batch, Data
from util.pdb_parsing import *
import pickle
from tqdm import tqdm
from pathlib import Path
import time
from torch import Tensor
from typing import List, Dict, Tuple


class DataSet(torch.utils.data.Dataset):

    def __init__(self, data_path:Path) -> None:
        """
        Initialize the dataset by loading pickled graph data from a directory.

        Args:
            data_path (Path): Path object pointing to a directory containing .pkl files.
        """
        self.fs = []
        start_time = time.time()
        for pkl in list(data_path.iterdir()): 
            with open(pkl, 'rb') as f: 
                data = pickle.load(f)
                self.fs.append(data)
        end_time = time.time() 
        print("Loading Time : ", int(end_time - start_time))
        
    def __len__(self) -> int:
        return len(self.fs)
        
    def __getitem__(self, index:int) -> Data:
        ipdb = index%len(self.fs)
        G = self.fs[ipdb]
        return G
    

def collate(samples:List[Data]) -> Data: 
    """
    Custom collate function for PyTorch Geometric DataLoader.

    Args:
        samples (List[Data]): List of individual graph samples to be batched.

    Returns:
        Data: A single batched Data object with updated `pepidx` and `pepidx2` fields.
    """
    bG = Batch.from_data_list(samples) 

    pepidx_list = []  # Global
    pepidx2_list = [] # Local
    offset = 0
    for i,data in enumerate(samples): 
        pepidx_list.append(data.pepidx + offset)
        pepidx2_list.append(data.pepidx.tolist())
        offset += data.num_nodes 
    bG.pepidx = torch.cat(pepidx_list) # Combined global peptide indices
    bG.pepidx2 = pepidx2_list          # List of local peptide indices (per graph)

    return bG