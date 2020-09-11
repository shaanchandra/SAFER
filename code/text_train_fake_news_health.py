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
from text_train_main import *


                
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required Paths
    parser.add_argument('--data_path', type = str, default = '../data',
                          help='path to dataset folder that contains the folders to gossipcop or politifact folders (raw data)')
    parser.add_argument('--glove_path', type = str, default = '../data/glove/glove.840B.300d.txt',
                          help='path for Glove embeddings (850B, 300D)')
    parser.add_argument('--model_checkpoint_path', type = str, default = '../model_checkpoints',
                          help='Directory for saving trained model checkpoints')
    parser.add_argument('--vis_path', type = str, default = '../vis_checkpoints',
                          help='Directory for saving tensorboard checkpoints')
    parser.add_argument("--model_save_name", type=str, default= 'best_model.pt',
                       help = 'saved model name')

    #### Training Params ####
    
    # Named params    
    parser.add_argument('--data_name', type = str, default = 'HealthStory',
                          help='dataset name:  HealthStory / HealthRelease')
    parser.add_argument('--model_name', type = str, default = 'cnn',
                          help='model name: bilstm / bilstm_pool / bilstm_reg / han / cnn / \
                              bert-base-cased / bert-large-cased / xlnet-base-cased / xlnet-large-cased / roberta-base / roberta-large')
    parser.add_argument('--embed_name', type = str, default = 'glove',
                          help='type of word embeddings used: glove/ elmo/ bert/ xlnet / roberta"')
    parser.add_argument('--optimizer', type = str, default = 'Adam',
                        help = 'Optimizer to use for training')
    parser.add_argument('--loss_func', type = str, default = 'bce',
                        help = 'Loss function to use for optimization: bce / bce_logits / ce')
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
    parser.add_argument('--n_classes', type = int, default = 3,
                          help='number of classes"')    
    parser.add_argument('--pos_wt', type = float, default = 0.3,
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
    parser.add_argument('--wdrop', type = float, default = 0.2,
                        help = 'Regularization - weight dropout')
    parser.add_argument('--embed_drop', type = float, default = 0.1,
                        help = 'Regularization - embedding dropout')
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
                      help='Whether to fine-tune BERT or not (freeze = True will not fine-tune)"')
    
    # Transformer Model params
    parser.add_argument('--output_dir_transf', type=str, default="transf_outputs/",
                        help='directory to store all outputs (model checkpoints and evaluation results)')
    parser.add_argument('--cache_dir_transf', type=str, default="transf_cache/",
                        help='directory to store all cache (pre-trained wts, features, etc')
    parser.add_argument('--classifier', type=str, default="mlp",
                        help='Option of cnn / mlp. If CNN then Roberta is used as word embeddings generator for CNN')
    parser.add_argument('--max_seq_length_transf', type=int, default=512,
                        help='max no. of tokens to process with Transformer models')
    parser.add_argument('--train_batch_size_transf', type=int, default=8,
                        help='batch size to use while training')
    parser.add_argument('--eval_batch_size_transf', type=int, default=8,
                        help='batch size to use while evaluating')
    parser.add_argument('--gradient_accumulation_steps_transf', type=int, default=1,
                        help='The number of training steps to execute before performing a optimizer.step(). Effectively increases \
                            the training batch size while sacrificing training time to lower memory consumption.')
    parser.add_argument('--num_train_epochs_transf', type=int, default=1,
                        help='no. of training epochs')
    parser.add_argument('--weight_decay_transf', type=float, default=0.5,
                        help='Weight Decay for AdamW')
    parser.add_argument('--adam_epsilon_transf', type=float, default=1e-8,
                        help='Epsilon for AdamW')
    parser.add_argument('--learning_rate_transf', type=float, default=1e-5,
                        help='Learnign Rate for training')
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.5,
                        help='The dropout for all fully connected layers in the embeddings, encoder, and pooler. (default is 0.1)')
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.5,
                        help='The dropout ratio for the attention probabilities.(default is 0.1)')
    parser.add_argument('--warmup_ratio_transf', type=float, default=0.06,
                        help='Ratio of total_steps to use as warm-up by lr_Scheduler')    
    parser.add_argument('--warmup_steps_transf', type=int, default=0,
                        help='No.of of total_steps to use as warm-up by lr_Scheduler (overwrites warmup_ratio_transf')
    parser.add_argument('--max_grad_norm_transf', type=float, default=1,
                        help='Maximum gradient clipping.')
    parser.add_argument('--do_lower_case_transf', type=bool, default=False,
                        help='Set to True when using uncased models')
    parser.add_argument('--evaluate_during_training_transf', type=bool, default=True,
                        help='Set to True to perform evaluation while training models (within an epoch)')
    parser.add_argument('--evaluate_during_epoch_ratio_transf', type=float, default= 2,
                        help='Perform evaluaiton on val set these many equally paced times (within an epoch)')
    parser.add_argument('--evaluate_during_training_verbose_transf', type=bool, default=True,
                        help='Print results from evaluation during training.')
    parser.add_argument('--use_cached_eval_features_transf', type=bool, default=True,
                        help='Evaluation during training uses cached features. Setting this to False will cause features\
                            to be recomputed at every evaluation step.')
    parser.add_argument('--save_eval_checkpoints_transf', type=bool, default=False,
                        help='Save a model checkpoint for every evaluation performed.')
    parser.add_argument('--logging_steps_transf', type=int, default= 20,
                        help='Log training loss and learning at every specified number of steps.')
    parser.add_argument('--save_steps_transf', type=int, default=0,
                        help='Save a model checkpoint at every specified number of steps.')
    parser.add_argument('--no_cache_transf', type=bool, default=True,
                        help='Cache features to disk.')
    parser.add_argument('--save_model_every_epoch_transf', type=bool, default=False,
                        help='Save a model at the end of every epoch.')
    parser.add_argument('--tensorboard_dir_transf', type=str, default='./transf_vis',
                        help='Save a model at the end of every epoch.')
    parser.add_argument('--overwrite_output_dir_transf', type=bool, default=True,
                        help='If True, the trained model will be saved to the ouput_dir and will \
                            overwrite existing saved models in the same directory.')
    parser.add_argument('--reprocess_input_data_transf', type=bool, default=True,
                        help='If True, the input data will be reprocessed even if a cached file \
                            of the input data exists in the cache_dir.')    
    parser.add_argument('--n_gpu_transf', type=int, default=4,
                        help='No. of GPUs to use')
    parser.add_argument('--silent_transf', type=bool, default=True,
                        help='Disables progress bars.')
    parser.add_argument('--use_early_stopping_transf', type=bool, default=True,
                        help='Use early stopping to stop training when eval_loss does not improve \
                            (based on early_stopping_patience, and early_stopping_delta)')
    parser.add_argument('--early_stopping_patience_transf', type=int, default=6,
                        help='Terminate training after these many evaluations without an improvement \
                            in eval_loss greater then early_stopping_delta.')
    parser.add_argument('--early_stopping_delta_transf', type=float, default=1e-3,
                        help = 'The improvement over best_eval_loss necessary to count as a better checkpoint')
    parser.add_argument('--manual_seed_transf', type=int, default=21,
                        help = 'Seed for training for reproduction')
    parser.add_argument('--use_cuda', type=bool, default=True,
                        help = 'Whethere to use GPU or not')
    parser.add_argument('--extract_embeddings', type=bool, default=False,
                        help = "To extract seq embedding [CLS] and apply attention on parts of the doc before classification")
    parser.add_argument('--sliding_window', type=bool, default=False,
                        help = 'Whether to iterate whole document based on a sliding window stride')
    parser.add_argument('--stride_transf', type=int, default=0.99,
                        help = 'stride * max_seq_len will be slided over')

    
    


    args, unparsed = parser.parse_known_args()
    config = args.__dict__

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device    

    if config['data_name'] == 'pheme':
        config['n_classes'] = 3
    else:
        config['n_classes'] = 1            
   
    # Check all provided paths:    
    if config['embed_name'] not in ['bert', 'xlnet', 'roberta']:
        config['model_path'] = os.path.join(config['model_checkpoint_path'], config['data_name'], config['embed_name'], config['model_name'])
        config['vis_path'] = os.path.join(config['vis_path'], config['data_name'], config['embed_name'], config['model_name'])
    else:
        config['model_path'] = os.path.join(config['output_dir_transf'], config['data_name'], config['embed_name'], config['model_name'])
        config['vis_path'] = os.path.join(config['tensorboard_dir_transf'], config['data_name'], config['embed_name'], config['model_name'])
        
    if not os.path.exists(config['data_path']):
        raise ValueError("[!] ERROR: Dataset path does not exist")
    else:
        print("\nData path checked..")
        config['data_path'] = os.path.join(config['data_path'], config['data_name'])
    if not os.path.exists(config['glove_path']):
        raise ValueError("[!] ERROR: Glove Embeddings path does not exist")
    else:
        print("\nGLOVE embeddings path checked..")
    if config['embed_name'] not in ['bert', 'xlnet', 'roberta'] and not os.path.exists(config['model_path']):
        print("\nCreating checkpoint path for saved models at:  {}\n".format(config['model_path']))
        os.makedirs(config['model_path'])
    else:
        print("\nModel save path checked..")
    if config['data_name'] not in ['HealthStory', 'HealthRelease']:
        raise ValueError("[!] ERROR:  data_name is incorrect. This file is for trianin of: HealthStory / HealthRelease  only !")
    else:
        print("\nData name checked...")
    if config['model_name'] not in ['bilstm', 'bilstm_pool', 'bilstm_reg', 'han', 'cnn', 'bert-base-cased' , 'bert-large-cased' , 'xlnet-base-cased' , 'xlnet-large-cased', 'roberta-base', 'roberta-large']:
        raise ValueError("[!] ERROR:  model_name is incorrect. Choose one of - bilstm / bilstm_pool / bilstm_reg / han / cnn\
                         bert-base-cased / bert-large-cased / xlnet-base-cased / xlnet-large-cased / roberta-base / roberta-large")
    else:
        print("\nModel name checked...")
    if not os.path.exists(config['vis_path']):
        print("\nCreating checkpoint path for Tensorboard visualizations at:  {}\n".format(config['vis_path']))
        os.makedirs(config['vis_path'])
    else:
        print("\nTensorbaord Visualization path checked..")
        print("Cleaning Visualization path of older tensorboard files...\n")
        # shutil.rmtree(config['vis_path'])

    
    
    if config['embed_name'] not in ['bert', 'xlnet', 'roberta']:
        train_args = None
        # Print args
        print("\n" + "x"*50 + "\n\nRunning training with the following parameters: \n")
        for key, value in config.items():
            if not key.endswith('transf'):
                print(key + ' : ' + str(value))
        print("\n" + "x"*50)
    
    else:
        # Create train_args for transformer models
        train_args = {}
        print("\n" + "x"*50 + "\n\nRunning TRANSFORMER training with the following parameters: \n")
        print("model_name : ", config['model_name'])
        print("hidden_dropout_prob : ", config['hidden_dropout_prob'])
        print("attention_probs_dropout_prob : ", config['attention_probs_dropout_prob'])
        for key, value in config.items():
            if key.endswith('transf'):
                new_key = key.split('_transf')[0]
                train_args[new_key] = value
                print(new_key + ' : ' + str(value))
        print("sliding_window : ", config['sliding_window'])
        print("\n" + "x"*50)
        train_args['tensorboard_dir'] = config['vis_path']
        train_args['output_dir'] = config['model_path']
        train_args['fp16'] = False
        if config['classifier'] == 'cnn':
            train_args['sliding_window'] = True
            config['freeze'] = True
            config['extract_embeddings'] = False
        
    config['writer'] = SummaryWriter(config['vis_path'])
        
    
    # Prepare the datasets and iterator for training and evaluation based on Glove or Elmo embeddings
    if config['embed_name'] == 'glove':
        config['train_loader'], config['dev_loader'], config['test_loader'], config['TEXT'], config['LABEL'], config['train_split'], config['val_split'], config['test_split'] = prepare_glove_training(config)
        config['vocab'] = config['TEXT'].vocab
    elif config['embed_name'] in ['bert', 'xlnet', 'roberta']:
        config['train_df'], config['val_df'], config['test_df'] = prepare_transformer_training(config)
    elif config['embed_name']=='elmo' and config['model_name']!='han':
        config['train_data'], config['train_label'], config['val_data'], config['val_labels'], config['test_data'], config['test_label'] = prepare_elmo_training(config)
        # print(len(train_labels))
    elif config['embed_name']=='elmo' and config['model_name']=='han':
        config['train_loader'], config['val_loader'], config['test_loader'] = prepare_HAN_elmo_training(config)
        
           
        
    
    try:
        doc_encoder = Doc_Encoder_Main(config, train_args)
        doc_encoder.train_main()
    except KeyboardInterrupt:
        print("Keyboard interrupt by user detected...\nClosing the tensorboard writer!")
        config['writer'].close()