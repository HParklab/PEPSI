from argparse import ArgumentParser
import torch

def set_arguments():
    parser = ArgumentParser()
    
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # dataset
    parser.add_argument('--pkl_path', type=str, default='/scratch/jsi0613/CG_data/pickles/')
    parser.add_argument('--t_dim', type=int, default=4, help='time embedding dimension')
    
    # model
    parser.add_argument('--lr', type=float, default=3.0e-4)
    parser.add_argument('--model_path', type=str, default='/home/jsi0613/projects/ddpm_coarse/weights/')
    parser.add_argument('--MAXEPOCHS', type=int, default=1000)
    parser.add_argument('--timestep', type=int, default=100)
    parser.add_argument('--model_name', type=str, default='CG_test')

    # sampling
    parser.add_argument('-sample_path', type=str, default='/home/jsi0613/projects/ddpm_coarse/data/samples/')
    
    args = parser.parse_args()
    
    return args