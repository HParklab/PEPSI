# PEPSI
A Generalizable Hierarchical All-Atom Peptide binder generative model

<img width="981" height="384" alt="스크린샷 2025-07-23 오후 3 14 09" src="https://github.com/user-attachments/assets/24b45f04-7dbb-4278-be06-f40dda66ed0e" />

## Table of Contents 

## Citiation 

## Installation
```bash
git clone https://github.com/HParklab/PEPSI.git
cd PEPSI

# 1. 새 환경 만들기
conda create -n PEPSI python=3.11 -y
conda activate PEPSI

# 2. PyTorch (CUDA 12.1) 설치 
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 4. 나머지 패키지 설치 (수정된 requirements.txt 기준)
pip install -r requirements.txt
```

## Usage
### Unzip pretrained model
```bash
gzip -d params/Coarse-Grained/best.pkl.gz
```
### sample
```bash
python scripts/sample_pretrained.py --pdb_path {pdbfile path} --pdbnum {pdbfile} 
```


