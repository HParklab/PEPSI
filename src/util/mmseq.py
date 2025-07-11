from Bio.PDB import PDBParser, PPBuilder
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO
from pathlib import Path 
from tqdm import tqdm

def split_chains_to_fastas_grouped(pdb_dir, chain_x_path, chain_a_path):
    pdb_dir = Path(pdb_dir)
    parser = PDBParser(QUIET=True)
    ppb = PPBuilder()
    
    records_X = []
    records_A = []

    for pdb_path in tqdm(list(pdb_dir.glob("*.pdb"))):
        structure = parser.get_structure(pdb_path.stem, pdb_path)
        chain_a_seq = ""

        for model in structure:
            for chain in model:
                seq = ''.join(str(pp.get_sequence()) for pp in ppb.build_peptides(chain))
                if not seq:
                    continue

                if chain.id == 'X':
                    record_id = f"{pdb_path.stem}_{chain.id}"
                    records_X.append(SeqRecord(Seq(seq), id=record_id, description=""))
                else:
                    chain_a_seq += seq  # chain A, B, C... 모두 하나의 seq로 합침

        if chain_a_seq:
            record_id = f"{pdb_path.stem}_nonX"
            records_A.append(SeqRecord(Seq(chain_a_seq), id=record_id, description=""))

    SeqIO.write(records_X, chain_x_path, "fasta")
    SeqIO.write(records_A, chain_a_path, "fasta")

# 사용 예시
split_chains_to_fastas_grouped(
    pdb_dir="your_pdb_dir_path",  # PDB 파일들이 들어있는 경로
    chain_x_path="chainX.fasta",
    chain_a_path="chainA.fasta"
)

pdb_dir = Path("/scratch", "jsi0613", "CG_data", "pdbs")
fasta_dir = Path("/scratch", "jsi0613", "CG_data", "fastas")
split_chains_to_fastas_grouped(
    pdb_dir=pdb_dir,
    chain_x_path=fasta_dir.joinpath("chainX", "chainX.fasta"),
    chain_a_path=fasta_dir.joinpath("chainA", "chainA.fasta")
)