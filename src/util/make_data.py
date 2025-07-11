import torch, pickle
import numpy as np
import shutil
from pathlib import Path 
from Bio.PDB import PDBParser, PDBIO, Model, Chain, Residue, Atom, NeighborSearch
from Bio.PDB.StructureBuilder import StructureBuilder
from tqdm import tqdm

#####################################################################################
def get_interacting_residues(pdb_path, binder_chain_id, cutoff=4.0):
    """
        find residues interacting with binder_chain_id
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)
    model = structure[0]

    # chain 분리
    binder_chain = model[binder_chain_id]
    receptor_atoms = []

    for chain in model:
        if chain.id != binder_chain_id:
            for res in chain:
                for atom in res:
                    if atom.element != 'H':
                        receptor_atoms.append(atom)

    binder_atoms = [
        atom for res in binder_chain for atom in res if atom.element != 'H'
    ]

    # NeighborSearch
    ns = NeighborSearch(receptor_atoms)

    interacting_b_residues = set()

    for atom in binder_atoms:
        neighbors = ns.search(atom.coord, cutoff)
        if neighbors:
            interacting_b_residues.add(atom.get_parent())  # Residue 객체

    # residue 번호만 추출
    result = sorted([res.id[1] for res in interacting_b_residues])

    return result

def group_consecutive(nums):
    """
        make inside list with consecutive elements : [   ] => [[], [], []]
    """
    if not nums:
        return []

    nums = sorted(nums)
    groups = [[nums[0]]]

    for n in nums[1:]:
        if n == groups[-1][-1] + 1:
            groups[-1].append(n)
        else:
            groups.append([n])
    
    return groups

def save_modified_chain_pdb(pdb_path, output_path, chain_id, keep_start, keep_end):
    """
        Rule of modifying pdb's chain IDs
        1. no change in receptor 
        2. residue < keep_start : chain ID == Y 
        3. keep_start <= residue <= keep_end : chain ID == X 
        4. residue >= keep_end : chain ID == X
        5. delete keep_end+1 & keep_start-1
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_path)
    model = structure[0]

    # 새로운 structure 생성
    builder = StructureBuilder()
    builder.init_structure("modified")
    builder.init_model(0)
    builder.init_seg("    ")

    for chain in model:
        if chain.id != chain_id:
            # 다른 체인은 그대로 추가
            builder.structure[0].add(chain.copy())
        else:
            # 대상 체인 처리
            chain_X = Chain.Chain("X")  # keep 영역
            chain_Y = Chain.Chain("Y")  # keep 이전
            chain_Z = Chain.Chain("Z")  # keep 이후

            for res in chain:
                res_id = res.id[1]
                copied_res = res.copy()

                # 삭제 조건: 바로 앞/뒤 residue 제거
                if res_id == keep_start - 1 or res_id == keep_end + 1:
                    continue

                # 조건 분기
                if keep_start <= res_id <= keep_end:
                    chain_X.add(copied_res)
                elif res_id < keep_start:
                    chain_Y.add(copied_res)
                else:  # res_id > keep_end
                    chain_Z.add(copied_res)

            if len(chain_Y):
                builder.structure[0].add(chain_Y)
            if len(chain_X):
                builder.structure[0].add(chain_X)
            if len(chain_Z):
                builder.structure[0].add(chain_Z)

    # 저장
    io = PDBIO()
    io.set_structure(builder.get_structure())
    io.save(str(output_path))

def has_chain_break(pdb_file, chain_id='X', cutoff=4.1):
    """
        check if binder chain has chain break
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pdb_struct', pdb_file)
    
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            
            ca_coords = []
            prev_resid = None
            for res in chain:
                if 'CA' in res:
                    # skip hetero/water
                    if res.id[0] != ' ':
                        continue
                    ca_coords.append((res.id[1], res['CA'].get_coord()))

            ca_coords = sorted(ca_coords, key=lambda x: x[0])  # sort by residue index
            for (i1, coord1), (i2, coord2) in zip(ca_coords, ca_coords[1:]):
                dist = np.linalg.norm(coord2 - coord1)
                if dist > cutoff:
                    return True  # Chain break detected

    return False  # No chain break

def save_pdb_with_interacting_chains_only(pdb_path, output_path, target_chain_id='X', cutoff=6.0):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_path)
    model = structure[0]

    if target_chain_id not in model:
        raise ValueError(f"Chain {target_chain_id} not found in {pdb_path}")

    target_chain = model[target_chain_id]
    target_atoms = [atom for residue in target_chain for atom in residue if atom.element != 'H']

    ns = NeighborSearch(target_atoms)

    # 상호작용하는 체인 ID 저장
    interacting_chains = set([target_chain_id])
    for chain in model:
        if chain.id == target_chain_id:
            continue
        for residue in chain:
            for atom in residue:
                if atom.element != 'H' and ns.search(atom.coord, cutoff):
                    interacting_chains.add(chain.id)
                    break
            if chain.id in interacting_chains:
                break

    # 새로운 structure에 interacting 체인만 추가
    builder = StructureBuilder()
    builder.init_structure("filtered")
    builder.init_model(0)
    builder.init_seg("    ")
    structure_out = builder.get_structure()

    for chain in model:
        if chain.id in interacting_chains:
            structure_out[0].add(chain.copy())

    io = PDBIO()
    io.set_structure(structure_out)
    io.save(str(output_path))
#####################################################################################

# """ restore PDB & chain from unique_interface.txt """
# unique_interface = Path("/ml", "HmapPPDB", "unique_interface.txt")
# rawpdbs_path = Path("/ml", "HmapPPDB", "rawpdbs")
# subjects = []
# with open(unique_interface, 'r') as f: 
#     lines = f.readlines()

#     for line in tqdm(lines): 
#         ids = line.split(' ')[0]
#         id1 = ids.split('_')[1]
#         id2 = ids.split('_')[2].split('.')[0]
#         if id1 == id2: type = "homo"
#         else: type = "hetero"

#         pdbs = line.split(' ')[1] 
#         pdbID = pdbs.split('_')[0] + ".pdb"
#         chain_rec = pdbs.split('_')[1]
#         chain_bin = pdbs.split('_')[2]

#         subjects.append((pdbID, chain_rec, chain_bin, type))
# with open('/scratch/jsi0613/subjects.pkl', 'wb') as f: 
#     pickle.dump(subjects, f)

# """ Specifying the binder region in the PDB structure """
# infos = []
# errors = []
# for subject in tqdm(subjects): 

#     pdbID = subject[0]
#     chain_rec = subject[1]
#     chain_bin = subject[2]
#     type = subject[3]

#     try:
#         results = get_interacting_residues(rawpdbs_path.joinpath(pdbID), chain_bin)
#         results = group_consecutive(results)
        
#         for i in range(len(results)): 
#             for j in range(i+1, len(results)-i):
#                 cropped_result = results[i:j]
#                 cropped_result = [x for group in cropped_result for x in group]
                
#                 length = cropped_result[-1] - cropped_result[0] + 1
#                 interaction_residues = len(cropped_result)
#                 interaction_ratio = interaction_residues/length 

#                 if 4<length<26 and interaction_ratio>0.75: 
#                     infos.append((pdbID, chain_bin, (cropped_result[0], cropped_result[-1])))
#     except:
#         errors.append(pdbID)
#         continue
# print(len(infos))
# print(errors, len(errors))

# """ store PDB with modified chain IDs """
# new_path = Path("/scratch", "jsi0613", "pdbs")
# save_errors = []
# for info in tqdm(infos): 
#     pdbID = info[0]
#     chainID = info[1]
#     keep_start = info[2][0]
#     keep_end = info[2][-1]

#     pdb_name = pdbID.split('.')[0] + '_' + chainID + '_' + str(keep_start) + '_' + str(keep_end) + '.pdb'

#     try:
#         save_modified_chain_pdb(
#             pdb_path=str(rawpdbs_path.joinpath(pdbID)),
#             output_path=str(new_path.joinpath(pdb_name)),
#             chain_id=chainID,
#             keep_start=keep_start,
#             keep_end=keep_end
#         )
#     except: 
#         save_errors.append(pdbID)
#         continue
# print(save_errors, len(save_errors))

# """ check chain breaks """
# new_path = Path("/scratch", "jsi0613", "pdbs")
# chain_break_pdbs = []
# for pdb in tqdm(list(new_path.iterdir())):
#     pdbID = pdb.stem + '.pdb'
#     check = has_chain_break(pdb)

#     if check: 
#         shutil.move(pdb, "/scratch/jsi0613/chain_breaks/")

data_path = Path("/scratch", "jsi0613", "CG_data", "errors", "interface")
new_path = Path("/scratch", "jsi0613", "CG_data", "pdbs")

for pdb in tqdm(list(data_path.iterdir())): 
    pdbID = pdb.stem + '.pdb'

    save_pdb_with_interacting_chains_only( 
        pdb_path = pdb, 
        output_path = new_path.joinpath(pdbID),
    )