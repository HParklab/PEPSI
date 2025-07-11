from argparse import ArgumentParser
import torch

def set_arguments():
    parser = ArgumentParser()
    
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # dataset
    parser.add_argument('--datapath', type=str, default='/ml/pepbdb/setall/', help='pdb파일 저장 디렉토리')
    parser.add_argument('--t_dim', type=int, default=4, help='time embedding')


    parser.add_argument('--unique_interface', type=str, default='/ml/HmapPPDB/unique_interface.txt', help='unique interface for pdb_by_chain')
    parser.add_argument('--pdb_by_chain_path', type=str, default='/ml/HmapPPDB/pdb_by_chain/')
    parser.add_argument('--graph_storage', type=str, default='/scratch/jsi0613/refined_peplen_8-25_interaction50/coords/coarse_graph_maker.pkl')
    
    # model
    parser.add_argument('--lr', type=float, default=1.0e-6, help='learning rate')
    parser.add_argument('--model_path', type=str, default='/home/jsi0613/projects/ddpm_coarse//weights/', help='model weights dirctory')
    parser.add_argument('--MAXEPOCHS', type=int, default=1000, help='max epochs')
    parser.add_argument('--timestep', type=int, default=100)
    parser.add_argument('--perturb_weight', type=int, default=0.0)

    # model name : AA_dataset_lr_timestep_layer_scalingfactor_perturbweight
    parser.add_argument('--model_name', type=str, default='CG_coords100', help='저장될 이름')

    # sampling
    parser.add_argument('-sample_path', type=str, default='/home/jsi0613/projects/ddpm_coarse/data/samples/', help='sample 저장 디렉토리')
    
    args = parser.parse_args()
    
    return args