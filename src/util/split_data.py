import random
from collections import defaultdict
from pathlib import Path

tsv_path = Path("/scratch/jsi0613/CG_data/fastas/chainA/mmseq/chainA_clu.tsv")
seed = 42

cluster_to_members = defaultdict(list)
with open(tsv_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        cluster_rep, member = line.strip().split('\t')
        cluster_rep = cluster_rep.split('_nonX')[0]
        member = member.split('_nonX')[0]
        cluster_to_members[cluster_rep].append(member)
    
    member_to_cluster = {}
    for cluster_id, members in cluster_to_members.items():
        for m in members:
            if m in member_to_cluster:
                raise ValueError(f"{m} appears in multiple clusters!")
            member_to_cluster[m] = cluster_id

#     # 3. 클러스터 단위 셔플 및 split
    cluster_ids = list(cluster_to_members.keys())
    random.seed(seed)
    random.shuffle(cluster_ids)

    n = len(cluster_ids)
    n_train = int(0.8 * n)
    n_valid = int(0.1 * n)

    train_clusters = cluster_ids[:n_train]
    valid_clusters = cluster_ids[n_train:n_train + n_valid]
    test_clusters  = cluster_ids[n_train + n_valid:]

    def collect_members(cluster_list):
        return [m for cid in cluster_list for m in cluster_to_members[cid]]

    train_set = collect_members(train_clusters)
    valid_set = collect_members(valid_clusters)
    test_set  = collect_members(test_clusters)

# 저장
pdb_path = Path("/scratch/jsi0613/CG_data/pdbs")
for pdb in pdb_path.iterdir():
    pdbID = pdb.stem
    if pdbID in train_set: 
        # shutil.move(pdb, pdb_path.joinpath("trainlist"))
        continue
    elif pdbID in valid_set:
        # shutil.move(pdb, pdb_path.joinpath("validlist"))
        continue
    elif pdbID in test_set: 
        # shutil.move(pdb, pdb_path.joinpath("testlist"))
        continue
    else: 
        print("ERROR!!!!!!!!!!!")
        print(pdbID)