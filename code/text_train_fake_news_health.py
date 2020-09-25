import argparse, time, datetime, shutil
import sys, os, glob, json, re
from pprint import pprint
import warnings
warnings.filterwarnings("ignore")
# from torchsummary import summary

import numpy as np
import pandas

import torch
import torchtext
from torch.utils.tensorboard import SummaryWriter

# from tensorboardX import SummaryWriter
# from nltk import word_tokenize
import nltk
nltk.download('punkt')
sys.path.append("..")

from models.model import *
from models.transformer_model import *
from utils.utils import *
from utils.data_utils_gnn import *
from utils.data_utils_txt import *
from utils.data_utils_hygnn import *
from text_train_main import *


                
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required Paths
    parser.add_argument('--data_path', type = str, default = './data',
                          help='path to main dataset folder that contains dataset prcoessed files')
    parser.add_argument('--glove_path', type = str, default = './data/glove/glove.840B.300d.txt',
                          help='path for Glove embeddings (850B, 300D)')
    parser.add_argument('--model_checkpoint_path', type = str, default = './model_checkpoints',
                          help='Directory for saving trained model checkpoints')
    parser.add_argument('--vis_path', type = str, default = './vis_checkpoints',
                          help='Directory for saving tensorboard checkpoints')
    parser.add_argument("--model_save_name", type=str, default= 'best_model.pt',
                       help = 'saved model name')

    #### Training Params ####
    
    # Named params    
    parser.add_argument('--data_name', type = str, default = 'HealthStory',
                          help='dataset name:  politifact / HealthRelease / HealthStory / gossipcop')
    parser.add_argument('--model_name', type = str, default = 'cnn',
                          help='model name: bilstm / bilstm_pool / bilstm_reg / han / cnn / \
                              dbert-base-uncased / xlnet-base-cased / xlnet-large-cased / roberta-base / roberta-large')
    parser.add_argument('--embed_name', type = str, default = 'glove',
                          help='type of word embeddings used: glove/ elmo/ dbert/ xlnet / roberta"')
    parser.add_argument('--optimizer', type = str, default = 'Adam',
                        help = 'Optimizer to use for training')
    parser.add_argument('--loss_func', type = str, default = 'bce_logits',
                        help = 'Loss function to use for optimization: bce / bce_logits / ce')
    parser.add_argument('--optimze_for', type = str, default = 'f1',
                        help = 'Optimize for what measure during training and early stopping: loss / f1')
    parser.add_argument('--scheduler', type = str, default = 'step',
                        help = 'The type of lr scheduler to use anneal learning rate: step/multi_step')
    
    # Dimensions/sizes params   
    parser.add_argument('--batch_size', type = int, default = 32,
                          help='batch size for training"')
    parser.add_argument('--embed_dim', type = int, default = 300,
                          help='dimension of word embeddings used(GLove) =300, for Elmo = 512/256/128"')
    parser.add_argument('--lstm_dim', type = int, default = 124,
                          help='dimen of hidden unit of LSTM/BiLSTM networks"')
    parser.add_argument('--word_lstm_dim', type = int, default = 50,
                          help='dimen of hidden unit of word-level attn GRU units of HAN"')
    parser.add_argument('--sent_lstm_dim', type = int, default = 50,
                          help='dimen of hidden unit of sentence-level attn GRU units of HAN"')
    parser.add_argument('--kernel-num', type=int, default=128,
                        help='number of each kind of kernel')
    parser.add_argument('--fc_dim', type = int, default = 64,
                          help='dimen of FC layer"')
    
    # Numerical params
    parser.add_argument('--n_classes', type = int, default = 2,
                          help='number of classes"')    
    parser.add_argument('--pos_wt', type = float, default = 3,
                          help='Loss reweighting for the positive class to deal with class imbalance')
    parser.add_argument('--lr', type = float, default = 1e-4,
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
    parser.add_argument('--beta_ema', type = float, default = 0.99,
                        help = 'Temporal Averaging smoothing co-efficient')
    parser.add_argument('--dropout', type = float, default = 0.2,
                        help = 'Regularization - dropout in LSTM cells')
    parser.add_argument('--kernel-sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--seed', type=int, default=21,
                        help='set seed for reproducability')
    parser.add_argument('--han_max_batch_size', type=int, default=8000,
                        help='max empiracally calculated no. of tokens in a batch that fit for HAN dynamic batching')
    parser.add_argument('--log_every', type=int, default=200,
                        help='Log stats in Tensorboard every x iterations (not epochs) of training')
    
    # Options params
    parser.add_argument('--parallel_computing', type=bool, default=True,
                        help='To run the model on multiple nodes') 
    parser.add_argument('--lowercase', type=bool, default=False,
                        help='whether to lowercase the tokens or not')
    parser.add_argument('--freeze', type = bool, default = True,
                      help='Whether to fine-tune transformer or not (freeze = True will not fine-tune)"')    
    


    args, unparsed = parser.parse_known_args()
    config = args.__dict__

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device    

    config['n_classes'] = 1            
   
    # Check all provided paths:
    config['model_path'] = os.path.join(config['model_checkpoint_path'], config['data_name'], config['embed_name'], config['model_name'])
    config['vis_path'] = os.path.join(config['vis_path'], config['data_name'], config['embed_name'], config['model_name'])
    
        
    if not os.path.exists(config['data_path']):
        raise ValueError("[!] ERROR: Dataset path does not exist")
    else:
        print("\nData path checked..")
        config['data_path'] = os.path.join(config['data_path'], config['data_name'])
    if not os.path.exists(config['glove_path']):
        raise ValueError("[!] ERROR: Glove Embeddings path does not exist")
    else:
        print("\nGLOVE embeddings path checked..")
    if not os.path.exists(config['model_path']):
        print("\nCreating checkpoint path for saved models at:  {}\n".format(config['model_path']))
        os.makedirs(config['model_path'])
    else:
        print("\nModel save path checked..")
    if config['data_name'] not in ['HealthStory', 'HealthRelease', 'gossipcop', 'politifact']:
        raise ValueError("[!] ERROR:  data_name is incorrect. This file is for trianin of: HealthStory / HealthRelease / gossipcop / politifact  only !")
    else:
        print("\nData name checked...")
    if config['model_name'] not in ['bilstm', 'bilstm_pool', 'bilstm_reg', 'han', 'cnn', 'dbert-base-uncased', 'xlnet-base-cased' , 'xlnet-large-cased', 'roberta-base', 'roberta-large']:
        raise ValueError("[!] ERROR:  model_name is incorrect. Choose one of - bilstm / bilstm_pool / bilstm_reg / han / cnn\
                         dbert-base-uncased / xlnet-base-cased / xlnet-large-cased / roberta-base / roberta-large")
    else:
        print("\nModel name checked...")
    if not os.path.exists(config['vis_path']):
        print("\nCreating checkpoint path for Tensorboard visualizations at:  {}\n".format(config['vis_path']))
        os.makedirs(config['vis_path'])
    else:
        print("\nTensorbaord Visualization path checked..")
        print("Cleaning Visualization path of older tensorboard files...\n")
        # shutil.rmtree(config['vis_path'])

    
    # Print args
    print("\n" + "x"*50 + "\n\nRunning training with the following parameters: \n")
    for key, value in config.items():
        if not key.endswith('transf'):
            print(key + ' : ' + str(value))
    print("\n" + "x"*50)
        
    config['writer'] = SummaryWriter(config['vis_path'])
        
    
    # Prepare the datasets and iterator for training and evaluation based on Glove or Elmo embeddings
    prep_data = Prepare_Dataset(config)
    if config['embed_name'] == 'glove':
        config['train_loader'], config['dev_loader'], config['test_loader'], config['TEXT'] = prep_data.prepare_glove_training()
    elif config['embed_name'] in ['dbert', 'xlnet', 'roberta']:
        config['train_loader'], config['val_loader'], config['test_loader'] = prep_data.prepare_transformer_training()
    elif config['embed_name']=='elmo' and config['model_name']!='han':
        config['train_data'], config['train_label'], config['val_data'], config['val_labels'], config['test_data'], config['test_labels'] = prep_data.prepare_elmo_training()
        # print(len(train_labels))
    elif config['embed_name']=='elmo' and config['model_name']=='han':
        config['train_loader'], config['val_loader'], config['test_loader'] = prep_data.prepare_HAN_elmo_training(config)
        
           

    try:
        doc_encoder = Doc_Encoder_Main(config)
        doc_encoder.train_main(cache=False)
    except KeyboardInterrupt:
        print("Keyboard interrupt by user detected...\nClosing the tensorboard writer!")
        config['writer'].close()