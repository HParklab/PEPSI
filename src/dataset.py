import torch
import numpy as np
from torch_geometric.data import Batch
from util.pdb_parsing import *
import itertools, pickle
from util.arguments import set_arguments
from tqdm import tqdm
import time
from torch import Tensor
from typing import List, Dict, Tuple


args = set_arguments()

class DataSet(torch.utils.data.Dataset):
    """
    pdb to dataset

    """
    def __init__(self, fs, args = set_arguments()) -> None:
        """
        initializing DataSet

        Args:
            fs (np.ndarray): trainlist, validlist
            args (function): datapth, dcut, peplen parsed
        """
        self.fs = fs
        # self.datapath = args.data_storage
        # self.graph_list = []
        # for f in self.fs: 
        #     Gs = self.read_pdb(self.datapath + f)
        #     self.graph_list = self.graph_list + Gs
        # start_time = time.time()
        # print('data: ', args.graph_storage)
        # with open(args.graph_storage, 'rb') as f: 
        #     self.graph_list = pickle.load(f)
        # end_time = time.time() 
        # print('Data is Loaded, Time : ', int(end_time - start_time))
        
    def __len__(self):
        return len(self.fs)
        
    def __getitem__(self, index:int):
        """
        read_pdb의 input을 만들고 read_pdb를 실행시켜 pdb를 graph로 변환

        Args:
            index (int): index 
        """
        ipdb = index%len(self.fs)
        G = self.fs[ipdb]
        
        return G
        
    # def read_pdb(self, pdb:str):
    #     """
    #     pdb파일을 읽고 graph로 변환

    #     Args:
    #         pdb (str): pdb 파일명
    #         pepcen (int): peptide의 center residue
    #         tag (str): pdb ID
    #     """
    #     xyz, aas, seqsep, pepidx, crs, atmtp, _ = get_property(pdb, args.atom_type)
    #     resnums = int(len(pepidx)/14)
    #     Gs = []
    #     for target_resnum in range(1,resnums-1):
    #         G,_ = make_graph(xyz, pepidx, seqsep, atmtp, args.dcut, args.atom_type, target_resnum)
    #         Gs.append(G)
    #     return Gs


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


# def collate(samples):
#     """
#     graph를 batch graph로 변환
#     batch graph로 합쳤을 때의 pepidx를 반영

#     Args:
#         samples : batch size에 맞게 묶인 graph list  
#     """
#     bG = Batch.from_data_list(samples)
#     for i in range(len(bG.pepidx)-1):
#         bG.pepidx[i+1] = bG.pepidx[i+1] + bG.pepidx[i][-1]+1
#         bG.target_idx[i+1] = bG.target_idx[i+1] + bG.pepidx[i][-1]+1
#     bG.pepidx = list(itertools.chain.from_iterable(bG.pepidx))
#     bG.target_idx = list(itertools.chain.from_iterable(bG.target_idx))

#     return bG