import numpy as np
import torch, sys, os, pickle, time, yaml
from pathlib import Path
from torch.utils import data
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_dir)
from dataset import DataSet, collate
from arguments import set_arguments
from util.diffusion import Diffusion
from util.training_utils import *
from models.egnn import EGNN


def main():
    
    # Load params from config
    args = set_arguments()
    config_file = args.project_path + "configs/" + args.model_name  + ".yaml"
    with open(config_file, "r") as f: 
        config = yaml.safe_load(f)
    loader_params = config["loader_params"]
    loader_params["collate_fn"] = collate 
    model_params = config["model_params"]
    model_params["device"] = args.device 
    
    train_dir = Path(args.pkl_path).joinpath("trainlist")
    valid_dir = Path(args.pkl_path).joinpath("validlist")
    
    # dataloader
    train_set = DataSet(train_dir)
    train_loader = data.DataLoader(train_set, worker_init_fn=lambda _: np.random.seed(), **loader_params)
    valid_set = DataSet(valid_dir)
    valid_loader = data.DataLoader(valid_set, worker_init_fn=lambda _: np.random.seed(), **loader_params)
    
    model, optimizer, start_epoch, train_loss, valid_loss = load_model(EGNN, model_params, args.model_name, args.model_path, args.device, args.lr)

    noiser = Diffusion

    for epoch in range(start_epoch, args.MAXEPOCHS):
        # training
        start_time = time.time()
        temp_loss = run_an_epoch(model, optimizer, train_loader, noiser, args.device, args.timestep, args.t_dim, True)
        for key in temp_loss:
            train_loss[key].append(temp_loss[key])
        # validation
        with torch.no_grad():
            temp_loss = run_an_epoch(model, optimizer, valid_loader, noiser, args.device, args.timestep, args.t_dim, False)
            for key in temp_loss:
                valid_loss[key].append(temp_loss[key])

        end_time = time.time()
        ttime = end_time - start_time
        print(
            "Epoch %d, t: %d, "
            "Train Loss : TL %7.4f L1 %7.4f L2 %7.4f | "
            "Valid Loss : TL %7.4f L1 %7.4f L2 %7.4f "
            % ( 
                epoch,
                int(ttime),
                np.mean(train_loss['total'][-1]),
                np.mean(train_loss['loss1'][-1]),
                np.mean(train_loss['loss2'][-1]),
                np.mean(valid_loss['total'][-1]),
                np.mean(valid_loss['loss1'][-1]),
                np.mean(valid_loss['loss2'][-1])
            )
        )
        
        save_model(epoch, model, optimizer, train_loss, valid_loss, args.model_path, args.model_name)
        
        
if __name__ =="__main__":
    main()