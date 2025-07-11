import torch, os, json
import numpy as np
from torch import Tensor
from typing import List, Dict, Tuple
import math
# from Bio import PDB
from Bio.PDB import PDBParser
# from Bio.PDB.DSSP import DSSP

def com2zero_np(xyz:np.ndarray, idx:List) -> np.ndarray:
    """
    xyz coordinate를 기준이 되는 index의 coordinate의 com이 0이 되도록 평행이동
    
    Args:
        xyz (np.array): 전체 원본 xyz coordinate
        idx (List): COM의 기준이 되는 index
    
    Returns:
        np.array: 바뀐 xyz coordinate
    """
    
    com = np.mean(xyz[idx], axis=0, keepdims=True)
    xyz = xyz - com
    
    return xyz, com

def com2zero_tensor(xyz:Tensor, idx:List) -> Tensor:
    """
    xyz coordinate를 기준이 되는 index의 coordinate의 com이 0이 되도록 평행이동
    
    Args:
        xyz (Tensor): 전체 원본 xyz coordinate
        idx (Tensor): COM의 기준이 되는 index
    
    Returns:
        Tensor: 바뀐 xyz coordinate
    """
    
    com = torch.mean(xyz[idx], dim=0)
    xyz = xyz - com
    
    return xyz

def find_dist_neighbors(dX:Tensor,dcut:float,mode) -> Tensor:
    """
    dcut보다 작은 값이 있는 dX의 index를 찾는 함수
    
    Args:
        dX (Tensor): 원점과의 거리
        dcut (float): 기준이 되는 값
    
    Returns:
        Tensor: dX의 index
    """
    if mode == 'one':
        D = dX + 1.0e-6
        u = torch.where(D<dcut)[0]
    
        return u
    
    elif mode == 'two':
        D = dX + 1.0e-6
        u,v = torch.where(D<dcut)

        return u,v

def aa3toindex(aa3:str) -> int:
    """
    dcut보다 작은 값이 있는 dX의 index를 찾는 함수
    
    Args:
        aa3 (str): Amino Acid type
    
    Returns:
        int: index
    """
    
    aa3list = ['ALA','CYS','ASP','GLU','PHE','GLY','HIS','ILE','LYS','LEU',
               'MET','ASN','PRO','GLN','ARG','SER','THR','VAL','TRP','TYR']
    if aa3 in aa3list:
        return aa3list.index(aa3)
    else:
        return -1
    
def cos_scheduling(timesteps:int, s:float=0.008) -> Tensor:
    """
    cosine scheduling을 하는 함수
    
    Args:
        timesteps (int): 전체 timesteps
        s (float): constant
    
    Returns:
        Tensor: beta schedule tensor
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    
    return torch.clip(betas, 0.0001, 0.9999)


def split_consecutive(lst):
    if not lst:
        return []

    result = []
    temp = [lst[0]]

    for i in range(1, len(lst)):
        # 현재 값과 이전 값이 연속되지 않으면 새로운 리스트 생성
        if lst[i] == lst[i-1] + 1:
            temp.append(lst[i])
        else:
            result.append(temp)
            temp = [lst[i]]

    result.append(temp)  # 마지막 그룹 추가
    return result
        

def dihedral_angle(pdb, main_atom, side_atom):

    atom_properties = {}
    with open(pdb, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if not line.startswith('ATOM'): continue 
            atom_name = line[12:16].strip()
            if atom_name != main_atom and atom_name != side_atom: continue

            res_num = line[22:27].strip()
            xyz = [float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())]           
            
            if res_num not in atom_properties.keys(): 
                atom_properties[res_num] = {}
            atom_properties[res_num][atom_name] = xyz

    angles = {}
    for i,res in enumerate(atom_properties.keys()): 
        if i == 0 or i == len(atom_properties)-1 : continue 

        try:
            res = int(res)
            xyz_CAm1 = np.array(atom_properties[str(res-1)][main_atom])
            xyz_CA = np.array(atom_properties[str(res)][main_atom])
            xyz_CAp1 = np.array(atom_properties[str(res+1)][main_atom])
            vector1 = xyz_CA - xyz_CAm1
            vector2 = xyz_CA - xyz_CAp1
            main_angle = angle(vector1, vector2)

            xyz_CB = np.array(atom_properties[str(res)][side_atom])
            vector3 = xyz_CA - xyz_CB 
            n_CA = np.cross(vector1, vector2) 
            side_angle = 90 - angle(vector3, n_CA)

            angles[main_angle] = side_angle
        except: 
            continue
    
    return angles

def angle(vector1, vector2): 

    cos_theta = np.dot(vector1, vector2)
    theta = cos_theta/(np.linalg.norm(vector1)*np.linalg.norm(vector2))
    degree = math.degrees(theta)

    return degree


def positional_encoding(seq_len, d_model):

    position = np.arange(seq_len)[:, np.newaxis]  
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))  
    
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)  
    pe[:, 1::2] = np.cos(position * div_term)  
    pe = torch.tensor(pe, dtype=torch.float)
    
    return pe


def aminoacid_types(input): 

    if input == [0,1,1,1,1,2,3,4,5,5,5,6,6,6]:
        result = 'Arg'

    elif input == [0,1,1,1,1,1,2,3,4,5,5,6,6,6]:
        result = 'His'
     
    elif input == [0,1,1,1,1,1,1,2,3,4,5,6,6,6]:
        result = 'Lys'
    
    elif input == [0,1,1,1,1,1,1,1,2,3,4,6,7,7]:
        result = 'Asp'
    
    elif input == [0,1,1,1,1,1,1,2,3,4,6,6,7,7]:
        result = 'Glu'
    
    elif input == [0,1,1,1,1,1,1,1,1,1,2,3,4,7]:
        result = 'Ser'

    elif input == [0,1,1,1,1,1,1,1,1,2,3,4,6,7]:
        result = 'Thr'
    
    elif input == [0,1,1,1,1,1,1,1,2,3,4,5,6,7]:
        result = 'Asn' 
        
    elif input == [0,1,1,1,1,1,1,2,3,4,5,6,6,7]:
        result = 'Gln'
    
    elif input == [0,1,1,1,1,1,1,1,1,1,2,3,4,8]: 
        result = 'Cys or Sec'
    
    elif input == [0,1,1,1,1,1,1,1,1,1,1,2,3,4]: 
        result = 'Gly or Ala'
    
    elif input == [0,1,1,1,1,1,1,1,1,2,3,4,6,6]:
        result = 'Pro or Val'
    
    elif input == [0,1,1,1,1,1,1,1,2,3,4,6,6,6]:
        result = 'Ile or Leu'
    
    elif input == [0,1,1,1,1,1,1,1,2,3,4,6,6,8]:
        result = 'Met'

    elif input == [0,1,1,1,1,2,3,4,6,6,6,6,6,6]:
        result = 'Phe'
    
    elif input == [0,1,1,1,2,3,4,6,6,6,6,6,6,7]:
        result = 'Tyr'

    elif input == [0,1,2,3,4,5,6,6,6,6,6,6,6,6]:
        result = 'Trp'
    
    else: result = 'ERROR!!!!!'
    
    return result
    

def aminoacid_types2(input): 

    if input == sorted([0,1,1,1,1,2,3,4,0,0,0,2,2,2]):
        result = 'Arg'

    elif input == sorted([0,1,1,1,1,1,2,3,4,0,0,2,2,2]):
        result = 'His'
     
    elif input == sorted([0,1,1,1,1,1,1,2,3,4,0,2,2,2]):
        result = 'Lys'
    
    elif input == sorted([0,1,1,1,1,1,1,1,2,3,4,2,3,3]):
        result = 'Asp'
    
    elif input == sorted([0,1,1,1,1,1,1,2,3,4,2,2,3,3]):
        result = 'Glu'
    
    elif input == sorted([0,1,1,1,1,1,1,1,1,1,2,3,4,3]):
        result = 'Ser'

    elif input == sorted([0,1,1,1,1,1,1,1,1,2,3,4,2,3]):
        result = 'Thr'
    
    elif input == sorted([0,1,1,1,1,1,1,1,2,3,4,0,2,3]):
        result = 'Asn' 
        
    elif input == sorted([0,1,1,1,1,1,1,2,3,4,0,2,2,3]):
        result = 'Gln'
    
    elif input == sorted([0,1,1,1,1,1,1,1,1,1,2,3,4,5]): 
        result = 'Cys or Sec'
    
    elif input == sorted([0,1,1,1,1,1,1,1,1,1,1,2,3,4]): 
        result = 'Gly or Ala'
    
    elif input == sorted([0,1,1,1,1,1,1,1,1,2,3,4,2,2]):
        result = 'Pro or Val'
    
    elif input == sorted([0,1,1,1,1,1,1,1,2,3,4,2,2,2]):
        result = 'Ile or Leu'
    
    elif input == sorted([0,1,1,1,1,1,1,1,2,3,4,2,2,5]):
        result = 'Met'

    elif input == sorted([0,1,1,1,1,2,3,4,2,2,2,2,2,2]):
        result = 'Phe'
    
    elif input == sorted([0,1,1,1,2,3,4,2,2,2,2,2,2,3]):
        result = 'Tyr'

    elif input == sorted([0,1,2,3,4,0,2,2,2,2,2,2,2,2]):
        result = 'Trp'
    
    else: result = 'ERROR!!!!!'
    
    return result

def batch_com(x, y, dimnum): 

    B = int(x.shape[0]/dimnum)

    for i in range(B):
        x[i*dimnum:(i+1)*dimnum] -= (x[i*dimnum:(i+1)*dimnum][1].clone() + x[i*dimnum:(i+1)*dimnum][4].clone())/2
        y[i*dimnum:(i+1)*dimnum] -= (y[i*dimnum:(i+1)*dimnum][0].clone() + y[i*dimnum:(i+1)*dimnum][2].clone() + y[i*dimnum:(i+1)*dimnum][3].clone())/3
    
    return x, y

def batch_com1(x, dimnum): 

    B = int(x.shape[0]/dimnum)

    for i in range(B):
        x[i*dimnum:(i+1)*dimnum] -= (x[i*dimnum:(i+1)*dimnum][1].clone() + x[i*dimnum:(i+1)*dimnum][4].clone())/2
      
    return x


def dist_map(x,idxs,q):

    distmaps = []
    for idx in idxs: 
        distmap = torch.sqrt(torch.sum((x[idx][None,:] - x[idx][:,None])**2 + 1e-8, dim=2)).unsqueeze(2)
        distmaps.append(distmap)
    if len(distmaps) == 0:
        distmaps = torch.zeros(q,q,1)
    else:
        distmaps = torch.cat(distmaps, dim=2)
    
    return distmaps

def dist_map1(x,idx):

    distmap = torch.sqrt(torch.sum((x[idx][None,:] - x[idx][:,None])**2 + 1e-7, dim=2))

    return distmap

def make_distmap_list(xyz, batch): 

    dist_maps = []

    for i in batch.unique(sorted=True):
        idx = (batch == i).nonzero(as_tuple=True)[0]
        coord = xyz[idx]
        dist = torch.cdist(coord,coord,p=2)
        dist_maps.append(dist)

    return dist_maps

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def noise_schedule(s1, sT, w, T):
    t = np.arange(T)
    s = (sT - s1) / (sigmoid(-w) - sigmoid(w))
    b = 0.5 * (s1 + sT - s)
    alpha_bars = s * sigmoid(-w * (2 * t / T - 1)) + b 

    return alpha_bars


def save_graph_as_pdb(node_xyz, node_attr, save_path):
    # atom type 이름
    atom_types = ['N', 'CA', 'C', 'O', 'CB', 'N', 'C', 'O', 'S']
    
    # tensor라면 numpy로 변환
    if isinstance(node_xyz, torch.Tensor):
        node_xyz = node_xyz.detach().cpu().numpy()
    if isinstance(node_attr, torch.Tensor):
        node_attr = node_attr.detach().cpu().numpy()
    
    pdb_lines = []
    current_residue_idx = 1  # Residue index 시작
    atom_idx = 1  # Atom index (PDB에서 1부터)

    for i in range(len(node_xyz)):
        one_hot = node_attr[i]
        atom_type_idx = one_hot.argmax()
        atom_name = atom_types[atom_type_idx]

        # 'N' atom이면 새로운 residue로
        if atom_type_idx == 0 and i != 0:
            current_residue_idx += 1
        
        x, y, z = node_xyz[i]

        pdb_line = (
            f"ATOM  {atom_idx:5d} {atom_name:^4} {'ALA'} A{current_residue_idx:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]:>2s}"
        )
        pdb_lines.append(pdb_line)
        atom_idx += 1

    # 파일 저장
    with open("/home/jsi0613/projects/ddpm_SC/data/graph2pdb/" + save_path, "w") as f:
        for line in pdb_lines:
            f.write(line + "\n")

    print(f"Saved to {save_path}")


def get_angle(a, b, c):
    """Compute angle (in degrees) formed by atoms a-b-c."""
    ba = a.coord - b.coord
    bc = c.coord - b.coord
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    # numerical error 방지
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def is_hbond(donor_atom, hydrogen_atom, acceptor_atom, distance_cutoff=3.5, angle_cutoff=30):
    """수소 결합 조건: 거리 + 각도 모두 만족해야 함"""
    distance = np.linalg.norm(hydrogen_atom.coord - acceptor_atom.coord)
    if distance > distance_cutoff:
        return False

    angle = get_angle(donor_atom, hydrogen_atom, acceptor_atom)  # donor-H-acceptor
    if angle > (180 - angle_cutoff):  # angle이 클수록 직선(좋은 h-bond)
        return True
    return False

def count_hbond_residue_pairs(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('complex', pdb_path)

    binder_atoms = []
    target_atoms = []

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if chain.id == 'X':  # binder
                        binder_atoms.append((residue, atom))
                    else:  # target
                        target_atoms.append((residue, atom))

    hbond_residue_pairs = set()

    # binder의 donor-hydrogen을 찾고, target의 acceptor를 찾음
    for b_res, b_atom in binder_atoms:
        if b_atom.element in ['N', 'O']:  # donor 가능 원자
            # b_res 안에서 b_atom에 연결된 수소 원자 찾기
            hydrogens = [atom for _, atom in binder_atoms if atom.get_parent() == b_res and atom.element == 'H']

            for h_atom in hydrogens:
                for t_res, t_atom in target_atoms:
                    if t_atom.element in ['N', 'O']:  # acceptor 가능 원자
                        if is_hbond(b_atom, h_atom, t_atom):
                            hbond_residue_pairs.add((b_res.get_id()[1]))

    return len(hbond_residue_pairs), hbond_residue_pairs


def check_hydrogens_in_chain(pdb_file, chain_id='X'):
    standard_residues = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU',
        'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE',
        'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
    }

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("model", pdb_file)

    model = structure[0]
    if chain_id not in model:
        print(f"Chain {chain_id} not found.")
        return

    chain = model[chain_id]
    all_ok = True
    for residue in chain:
        resname = residue.get_resname()
        if resname not in standard_residues:
            continue  # skip water, ligands, etc.

        has_h = any(atom.element == 'H' for atom in residue)
        if not has_h:
            print(f"Missing hydrogen in residue: {resname} {residue.get_id()}")
            all_ok = False

    if all_ok:
        ok = 1
    else:
        ok = 0 

    return ok
