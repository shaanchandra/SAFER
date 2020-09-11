import torch
import torch.nn.functional as F
import argparse, time, datetime, shutil
import sys, os, glob, json, random
import numpy as np
import warnings
warnings.filterwarnings("ignore")
sys.path.append("..")
# from torchsummary import summary

from scipy.sparse import csr_matrix, lil_matrix, save_npz, load_npz
from sklearn.metrics import accuracy_score, classification_report

from torch.utils.tensorboard import SummaryWriter

# from tensorboardX import SummaryWriter
# from nltk import word_tokenize
import nltk
nltk.download('punkt')

from models.gnn_model import Graph_Net, Relational_GNN
from models.transformer_model import *
from utils.utils import *
from gnn_train_main_mtl import Graph_Net_MTL_Main
from gnn_train_main import Graph_Net_Main







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required Paths
    parser.add_argument('--data_path', type = str, default = '../data/complete_data',
                          help='path to dataset folder that contains the adj and feat matrices, etc')
    parser.add_argument('--model_checkpoint_path', type = str, default = '../model_checkpoints_gnn',
                          help='Directory for saving trained model checkpoints')
    parser.add_argument('--vis_path', type = str, default = '../vis_checkpoints_gnn',
                          help='Directory for saving tensorboard checkpoints')
    parser.add_argument("--model_save_name", type=str, default= 'best_model_gnn_lr.pt',
                       help = 'saved model name')
    
    #### Training Params ####
    
    # Named params    
    parser.add_argument('--data_name', type = str, default = 'pheme',
                          help='dataset name: politifact / gossipcop / pheme / rumoreval / HealthStory / HealthRelease')
    parser.add_argument('--model_name', type = str, default = 'gcn',
                          help='model name: gcn / graph_sage / graph_conv / gat / rgcn / rsage / rgat')
    parser.add_argument('--training_setting', type = str, default = 'normal',
                          help='train in either : normal / mtl (stance + veracity)')
    parser.add_argument('--mode', type = str, default = 'lr',
                          help='Whether to train in transductive (normal) way or inductive way (lr)')
    parser.add_argument('--optimizer', type = str, default = 'Adam',
                        help = 'Optimizer to use for training')
    parser.add_argument('--loss_func', type = str, default = 'ce',
                        help = 'Loss function to use for optimization: bce / bce_logits / ce')
    parser.add_argument('--scheduler', type = str, default = 'step',
                        help = 'The type of lr scheduler to use anneal learning rate: step/multi_step')
    
    # Dimensions/sizes params   
    parser.add_argument('--batch_size', type = int, default = 2,
                          help='batch size for training"')
    parser.add_argument('--embed_dim', type = int, default = 300,
                          help='dimension of hidden layers of the graph network')
    parser.add_argument('--fc_dim', type = int, default = 64,
                          help='dimension of hidden layers of the MLP classifier')

    
    # Numerical params
    parser.add_argument('--num_rels', type = int, default = 4,
                          help='No. of types of edges present"')
    parser.add_argument('--clusters', type = int, default = 2,
                          help='No. of clusters of sub-graphs for cluster-GCN"')
    parser.add_argument('--pos_wt', type = float, default = 0.3,
                          help='Loss reweighting for the positive class to deal with class imbalance')
    parser.add_argument('--lr', type = float, default = 5e-3,
                          help='Learning rate for training')
    parser.add_argument('--weight_decay', type = float, default = 1e-3,
                        help = 'weight decay for optimizer')
    parser.add_argument('--momentum', type = float, default = 0.8,
                        help = 'Momentum for optimizer')
    parser.add_argument('--max_epoch', type = int, default = 50,
                        help = 'Max epochs to train for')
    parser.add_argument('--lr_decay_step', type = float, default = 3,
                        help = 'No. of epochs after which learning rate should be decreased')
    parser.add_argument('--lr_decay_factor', type = float, default = 0.8,
                        help = 'Decay the learning rate of the optimizer by this multiplicative amount')
    parser.add_argument('--patience', type = float, default = 6,
                        help = 'Patience no. of epochs for early stopping')
    parser.add_argument('--dropout', type = float, default = 0.2,
                        help = 'Regularization - dropout on hidden embeddings')
    parser.add_argument('--node_drop', type = float, default = 0.2,
                        help = 'Node dropout to drop entire node from a batch')
    parser.add_argument('--seed', type=int, default=21,
                        help='set seed for reproducability')
    parser.add_argument('--log_every', type=int, default=150,
                        help='Log stats in Tensorboard every x iterations (not epochs) of training')
    
    # Options params
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='whether to shuffle batches')
    parser.add_argument('--rand_split', type=bool, default=False,
                        help='whether to use random split data and not event wise split data')
    parser.add_argument('--parallel_computing', type=bool, default=False,
                        help='To run the model on multiple nodes') 
    parser.add_argument('--cluster', type=bool, default=True,
                        help='whether to apply graph clustering before batching')
    parser.add_argument('--full_graph', type=bool, default=False,
                        help='whether to process the entire graph without clustering or sampling')
    
   

    args, unparsed = parser.parse_known_args()
    config = args.__dict__

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device   

    if config['data_name'] == 'pheme':
        config['n_classes_veracity'] = 3
        config['n_classes_stance'] = 4
        config['loss_func'] = 'ce' 
        config['n_classes'] = 3 # 2 for filtered version (2 class version)
    
    
    # Check all provided paths:    
    config['model_path'] = os.path.join(config['model_checkpoint_path'], config['data_name'], config['model_name'])
    config['vis_path'] = os.path.join(config['vis_path'], config['data_name'], config['model_name'])
        
    if not os.path.exists(config['data_path']):
        raise ValueError("[!] ERROR: Dataset path does not exist")
    else:
        print("\nData path checked..")   
    if not os.path.exists(config['model_path']):
        print("\nCreating checkpoint path for saved models at:  {}\n".format(config['model_path']))
        os.makedirs(config['model_path'])
    else:
        print("\nModel save path checked..")
    if config['model_name'] not in ['gcn', 'graph_sage', 'graph_conv', 'gat', 'rgcn', 'rsage', 'rgat']:
        raise ValueError("[!] ERROR:  model_name is incorrect. Choose one of - gcn / graph_sage / graph_conv / gat / rgcn / rsage / rgat")
    else:
        print("\nModel name checked...")
    if not os.path.exists(config['vis_path']):
        print("\nCreating checkpoint path for Tensorboard visualizations at:  {}\n".format(config['vis_path']))
        os.makedirs(config['vis_path'])
    else:
        print("\nTensorbaord Visualization path checked..")
        print("Cleaning Visualization path of older tensorboard files...\n")
        shutil.rmtree(config['vis_path'])

    

    # Print args
    print("\n" + "x"*50 + "\n\nRunning training with the following parameters: \n")
    for key, value in config.items():
        print(key + ' : ' + str(value))
    print("\n" + "x"*50)

    
    # Seeds for reproduceable runs
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Prepare dataset and iterators for training
    fold =1
    config['fold'] = fold
    if config['training_setting'] == 'normal':
        config['loader'], config['vocab_size'], config['data'] = prepare_gnn_training(config, fold=fold)
    elif config['training_setting'] == 'mtl':
        config['loader_stance'], config['loader_veracity'], config['vocab_size'], config['data_stance'], config['data_veracity'] = prepare_gnn_training_mtl(config, fold=fold)
    

    for fold in [1,3,4,6,8]: # range(1,10) for 9 event version
        
        avg_test_recall, avg_test_prec, avg_test_acc = 0,0,0
        avg_test_macro_f1, avg_test_micro_f1, avg_test_f1_weighted = 0,0,0
        
        config['fold'] = fold
        # Seeds for reproduceable runs
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        random.seed(config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        if config['training_setting'] == 'normal':
            config['loader'], config['vocab_size'], config['data'] = prepare_gnn_training(config, fold=fold)
        elif config['training_setting'] == 'mtl':
            config['loader_stance'], config['loader_veracity'], config['vocab_size'], config['data_stance'], config['data_veracity'] = prepare_gnn_training_mtl(config, fold=fold)
            
        config['writer'] = SummaryWriter(config['vis_path'])
        
        try:
            graph_net = Graph_Net_Main(config)
            graph_net.train_main(cache=False)
        except KeyboardInterrupt:
            print("Keyboard interrupt by user detected...\nClosing the tensorboard writer!")
            # print("Best val f1 = ", graph_net.best_val_f1)
            config['writer'].close()