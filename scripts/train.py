import numpy as np
import torch, sys, os, pickle, time, copy
from torch.utils import data
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_dir)
from dataset import DataSet, collate
from arguments import set_arguments
from util.diffusion import Diffusion
from util.training_utils import *
from models.egnn_jsi import EGNN
from util.lr_scheduler import *



def main():

    model_params = {
                'in_node_nf': 11,
                'hidden_nf': 160,
                'out_node_nf': 11,
                'in_edge_nf': 4,
                'device': torch.device('cuda'),
                'n_layers': 8
                }

    loader_params = {
    'shuffle': True,
    'num_workers': 5,
    'pin_memory': True,
    'collate_fn': collate,
    'batch_size': 25,
    'pin_memory': False}
    
    # datasets 
    args = set_arguments()

    start_time = time.time()
    with open(args.graph_storage, 'rb') as f: 
        graph_list = pickle.load(f)
    end_time = time.time()

    print('Data is Loaded, Time : ', int(end_time - start_time)) 
    train_list = graph_list[:int(len(graph_list)*0.8)]
    valid_list = graph_list[int(len(graph_list)*0.8):]
    
    # dataloader
    train_set = DataSet(train_list)
    train_loader = data.DataLoader(train_set, worker_init_fn=lambda _: np.random.seed(), **loader_params)
    valid_set = DataSet(valid_list)
    valid_loader = data.DataLoader(valid_set, worker_init_fn=lambda _: np.random.seed(), **loader_params)
    
    model, optimizer, start_epoch, train_loss, valid_loss = load_model(EGNN, model_params, args=set_arguments())

    noiser = Diffusion

    for epoch in range(start_epoch, args.MAXEPOCHS):
        # training
        start_time = time.time()
        temp_loss = run_an_epoch(model,optimizer,train_loader,noiser,True,args=set_arguments())
        
        for key in temp_loss:
            train_loss[key].append(temp_loss[key])
        # validation
        with torch.no_grad():
            temp_loss = run_an_epoch(model,optimizer,valid_loader,noiser,False,args=set_arguments())
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
        
        save_model(epoch, model, optimizer, train_loss, valid_loss, args=set_arguments())
        
        
if __name__ =="__main__":
    main()