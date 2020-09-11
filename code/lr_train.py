import torch
import torch.nn.functional as F
from torch_geometric.datasets import Reddit
from torch_geometric.data import ClusterData, ClusterLoader, Data
from torch_geometric.nn import SAGEConv, GCNConv

import argparse, time, datetime, shutil
import sys, os, glob, json

import warnings
warnings.filterwarnings("ignore")
# from torchsummary import summary

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

from torch.utils.tensorboard import SummaryWriter

# from tensorboardX import SummaryWriter
# from nltk import word_tokenize
import nltk
nltk.download('punkt')
from torch.autograd import Variable
import torch.nn as nn
from statistics import stdev
import sys
sys.path.append("..")

from models.model import *
from utils.utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





class LR_model(torch.nn.Module):
    def __init__(self, config):
        super(LR_model, self).__init__()
        if config['mode'] == 'gnn':
            self.in_dim = 256 if config['model_name'] != 'gat' else 300
            self.in_dim = self.in_dim*3 if config['model_name'] == 'gat' else self.in_dim
        elif config['mode'] == 'text':
            self.in_dim = 300
        else:
            self.in_dim = 256 if config['model_name'] != 'gat' else 300
            self.in_dim = self.in_dim*3 if config['model_name'] == 'gat' else self.in_dim
            self.in_dim+=1024
        self.classifier = nn.Linear(self.in_dim, config['n_classes'])
            

    def forward(self, x):
        out = self.classifier(x)
        return out
    

class LR_Learner():
    def __init__(self, config):
        self.best_val_acc, self.best_val_f1, self.best_val_recall, self.best_val_precision = 0, 0, 0, 0
        self.actual_best_f1 = 0
        self.preds_list, self.labels_list = [] , []
        self.train_f1, self.train_precision, self.train_recall, self.train_accuracy = 0,0,0,0
        self.train_loss = []
        self.threshold = 0
        self.prev_val_loss, self.not_improved  = 0, 0
        self.best_val_loss = 1e4
        self.total_iters = 0
        self.terminate_training = False
        self.start_epoch, self.iters = 1, 0
        self.preds, self.loss = 0, 0
        self.start = time.time()
         
        self.model = LR_model(config)
        if config['optimizer'] == 'Adam':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = config['lr'], weight_decay = config['weight_decay'])
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr = config['lr'], momentum = config['momentum'], weight_decay = config['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config['lr_decay_step'], gamma = config['lr_decay_factor'])
        self.criterion = nn.BCELoss() if config['loss_func'] == 'bce' else nn.CrossEntropyLoss()
        
    
    
    def eval_lr(self, test = False):
        self.model.eval()
        preds_list, labels_list = [], []
        docs_list = []
        eval_loss = []
        
        loader = test_loader if test else val_loader
        with torch.no_grad():
            for iters, (batch_x, batch_y, doc) in enumerate(loader):
                batch_x = Variable(batch_x)
                batch_y = Variable(batch_y)         
                preds = self.model(batch_x)
                if config['loss_func'] == 'bce':
                    preds = F.sigmoid(preds)
                labels = batch_y.long() if config['data_name'] == 'pheme' else batch_y.float()
                # labels = batch_y.float()
                preds = torch.where(torch.isnan(preds), torch.zeros_like(preds), preds)
                loss = self.criterion(preds.to(device), labels.to(device))
                eval_loss.append(loss.detach().item())
                if config['loss_func'] == 'ce':
                    preds = F.softmax(preds, dim=1)
                    preds = torch.argmax(preds, dim=1)
                else:
                    preds = (F.sigmoid(preds)>0.5).type(torch.FloatTensor) if config['loss_func'] == 'bce_logits' else (preds>0.5).type(torch.FloatTensor)
                preds_list.append(preds.cpu().detach().numpy())
                labels_list.append(labels.cpu().detach().numpy())
                docs_list.append(doc)

                
            preds_list = [pred for batch_pred in preds_list for pred in batch_pred]
            labels_list = [label for batch_labels in labels_list for label in batch_labels]
            docs_list = [doc for docs in docs_list for doc in docs]
            # if test:
            #     self.save_correct_preds(docs_list, preds_list, labels_list)
            eval_loss = sum(eval_loss)/len(eval_loss)
            
            if test:
                # target_names = ['False', 'True', 'Unverif']
                # print(classification_report(np.array(labels_list), np.array(preds_list), target_names=target_names))
                print(classification_report(np.array(labels_list), np.array(preds_list)))
            
            if not test:
                if config['data_name'] != 'pheme':
                    eval_f1, eval_macro_f1, eval_recall, eval_precision, eval_accuracy = evaluation_measures(config, np.array(preds_list), np.array(labels_list))
                    return eval_f1, eval_macro_f1, eval_precision, eval_recall, eval_accuracy, eval_loss
                else:
                    eval_f1, eval_recall, eval_precision, eval_accuracy = evaluation_measures_pheme(config, np.array(preds_list), np.array(labels_list))
                    return eval_f1, eval_precision, eval_recall, eval_accuracy, eval_loss  
                    # eval_f1, eval_f1_neg, eval_macro_f1, eval_recall, eval_precision, eval_accuracy = evaluation_measures_pheme_filtered(config, np.array(preds_list), np.array(labels_list))
                    # return eval_f1, eval_f1_neg, eval_macro_f1, eval_precision, eval_recall, eval_accuracy, eval_loss
            
            else:
                if config['data_name'] != 'pheme':
                    eval_f1, eval_macro_f1, eval_recall, eval_precision, eval_accuracy = evaluation_measures(config, np.array(preds_list), np.array(labels_list))
                    return eval_f1, eval_macro_f1, eval_precision, eval_recall, eval_accuracy, eval_loss
                else:
                    eval_f1, eval_recall, eval_precision, eval_accuracy = evaluation_measures_pheme(config, np.array(preds_list), np.array(labels_list))
                    return preds_list, labels_list, eval_f1, eval_precision, eval_recall, eval_accuracy, eval_loss  
            
    
    
    
    
    def save_correct_preds(self, docs, preds, labels):
        print(len(preds), len(labels), len(docs))
        idx_correct_preds = [idx for idx in range(len(preds)) if preds[idx]==labels[idx]]
        correct_pred_docs = [docs[idx] for idx in range(len(docs)) if idx in idx_correct_preds]
        print(len(correct_pred_docs))
        temp_dict = {'correct_preds' : list(correct_pred_docs)}
        model = 'roberta' if config['mode'] == 'text' else config['model_name']
        dataname = 'pheme_cv' if config['data_name'] == 'pheme' else config['data_name']
        if config['data_name'] != 'pheme':
            correct_doc_file = os.path.join('data', 'complete_data', dataname, 'cached_embeds', 'correct_docs_{}.json'.format(model))
        else:
            correct_doc_file = os.path.join('data', 'complete_data', dataname, 'fold_{}'.format(fold), 'correct_docs_{}.json'.format(model))
        print("Saving the list of correct test preds in : ", correct_doc_file)
        with open(correct_doc_file, 'w+') as j:
            json.dump(temp_dict, j)
        
        
    def train_epoch_step(self):
        self.model.train()
        lr = self.scheduler.get_lr()[0]
        self.total_iters += self.iters
        self.preds_list = [pred for batch_pred in self.preds_list for pred in batch_pred]
        self.labels_list = [label for batch_labels in self.labels_list for label in batch_labels]
        
        if config['data_name'] != 'pheme':
            self.train_f1, self.train_macro_f1, self.train_recall, self.train_precision, self.train_accuracy = evaluation_measures(config, np.array(self.preds_list), np.array(self.labels_list))
        else:
            self.train_f1, self.train_recall, self.train_precision, self.train_accuracy = evaluation_measures_pheme(config, np.array(self.preds_list), np.array(self.labels_list))
            # self.train_f1, self.train_f1_neg, self.train_macro_f1, self.train_recall, self.train_precision, self.train_accuracy = evaluation_measures_pheme_filtered(config, np.array(self.preds_list), np.array(self.labels_list))
            
        # Evaluate on dev set
        if config['data_name'] != 'pheme':
            self.eval_f1, self.eval_macro_f1, self.eval_precision, self.eval_recall, self.eval_accuracy, self.eval_loss = self.eval_lr()
        else:
            self.eval_f1, self.eval_precision, self.eval_recall, self.eval_accuracy, self.eval_loss = self.eval_lr()
            # self.eval_f1, self.eval_f1_neg, self.eval_macro_f1, self.eval_precision, self.eval_recall, self.eval_accuracy, self.eval_loss = self.eval_lr()
               
        # # print stats
        # if config['data_name'] != 'pheme':
        #     print_stats(config, self.epoch, self.train_loss, self.train_accuracy, self.train_f1, self.train_precision, self.train_recall,
        #                 self.eval_loss, self.eval_accuracy, self.eval_f1, self.eval_precision, self.eval_recall, self.start, lr)
        # else:
        #     print_stats_pheme(config, self.epoch, self.train_loss, self.train_accuracy, self.train_f1, self.train_precision, self.train_recall,
        #                 self.eval_loss, self.eval_accuracy, self.eval_f1, self.eval_precision, self.eval_recall, self.start, lr)
        
        
        
        # this_f1 = self.eval_f1[1] if config['data_name']=='pheme' else self.eval_macro_f1
        this_f1 =  self.eval_f1
        current_best = self.best_val_f1 
        # this_f1 =  self.eval_macro_f1
        # current_best = self.best_val_f1[1] if (not isinstance(self.best_val_f1, int) and config['data_name']=='pheme') else self.best_val_f1
        if this_f1 > current_best:
            print("New High Score! Saving model...")
            self.best_val_f1 = self.eval_f1
            # self.best_val_f1 = self.eval_macro_f1
            # self.best_val_f1 = this_f1
            self.best_val_acc = self.eval_accuracy
            self.best_val_recall = self.eval_recall
            self.best_val_precision = self.eval_precision
            best_model = self.model.state_dict()
            # Save the state and the vocabulary
            torch.save({
                'epoch': self.epoch,
                'best_val_f1' : self.best_val_f1,
                'model_state_dict': best_model,
                # 'model_classif_state_dict': model.classifier.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, os.path.join(model_path, config['model_save_name']))
            
        
        if self.epoch==1:
            print("Saving model !")
            best_model = self.model.state_dict()
            # Save the state and the vocabulary
            torch.save({
                'epoch': self.epoch,
                'best_val_f1' : self.best_val_f1,
                'model_state_dict': best_model,
                # 'model_classif_state_dict': model.classifier.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, os.path.join(model_path, config['model_save_name']))
            
            
        self.scheduler.step()
            
        
        # current_best = self.best_val_f1[1] if (not isinstance(self.best_val_f1, int) and config['data_name']=='pheme') else self.best_val_f1
        this_f1 =  self.eval_f1
        current_best = self.best_val_f1 
        
        if this_f1 - current_best!=0 and this_f1 - current_best < 1e-3:
            self.not_improved+=1
            print(self.not_improved)
            if self.not_improved >= config['patience']:
                self.terminate_training= True
        else:
            self.not_improved = 0
        
        if this_f1 > current_best and this_f1 - current_best > 1e-3:
            self.best_val_f1 = self.eval_f1
            self.not_improved=0        
        
        # if self.best_val_loss - self.eval_loss < 1e-3:
        #     self.not_improved+=1
        #     print(self.not_improved)
        #     if self.not_improved >= config['patience']:
        #         self.terminate_training= True
        # else:
        #     self.not_improved = 0
        
        # if self.eval_loss < self.best_val_loss and self.best_val_loss - self.eval_loss > 1e-3:
        #     self.best_val_loss = self.eval_loss
        #     self.not_improved = 0
            
            
        self.preds_list = []
        self.labels_list = []
        
        
        
        
    def train_iters_step(self):
        if config['loss_func'] == 'bce':
            self.preds = F.sigmoid(self.preds)
        
        self.batch_labels = self.batch_y.long() if config['data_name']=='pheme' else self.batch_y.float()
        # self.batch_labels = self.batch_y.float()
        # self.preds = torch.where(torch.isnan(self.preds), torch.zeros_like(self.preds), self.preds)
        self.loss = self.criterion(self.preds.to(device),  self.batch_labels.to(device))

        self.optimizer.zero_grad()
        self.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.optimizer.step()

        if config['loss_func'] == 'ce':
            self.preds = F.softmax(self.preds, dim=1)
            self.preds = torch.argmax(self.preds, dim=1)
        elif config['loss_func'] == 'bce':
            self.preds = (self.preds>0.5).type(torch.FloatTensor)
        elif config['loss_func'] == 'bce_logits': 
            self.preds = F.sigmoid(self.preds)
            self.preds = (self.preds>self.threshold).type(torch.FloatTensor).squeeze(1)
            
        self.preds_list.append(self.preds.cpu().detach().numpy())
        self.labels_list.append(self.batch_labels.cpu().detach().numpy())

        self.train_loss.append(self.loss.detach().item())
        
            
            
    def train_main(self):
        print("\n\n"+ "="*100 + "\n\t\t\t\t\t Training Network\n" + "="*100)

        # Seeds for reproduceable runs
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        random.seed(config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        print("\nBeginning training at:  {} \n".format(datetime.datetime.now()))
        
        for self.epoch in range(1, config['max_epoch']+1):
            for self.iters, (batch_x, batch_y, doc) in enumerate(train_loader):
                self.batch_x = Variable(batch_x)
                self.batch_y = Variable(batch_y)
                self.preds = self.model(self.batch_x)
                # print(self.preds)
                self.train_iters_step()
            self.train_epoch_step()
            if self.terminate_training:
                    break

        # Termination message
        if self.terminate_training:
            print("\n" + "-"*100 + "\nTraining terminated early because the Validation loss did not improve for   {}   epochs" .format(config['patience']))
        else:
            print("\n" + "-"*100 + "\nMaximum epochs reached. Finished training !!")
        
        print("\n" + "-"*50 + "\n\t\tEvaluating on test set\n" + "-"*50)
        
        if os.path.isfile(os.path.join(model_path, config['model_save_name'])):
            checkpoint = torch.load(os.path.join(model_path, config['model_save_name']))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            raise ValueError("No Saved model state_dict found for the chosen model...!!! \nAborting evaluation on test set...".format(config['model_name']))
            
        
        # Evaluate on dev set
        if config['data_name'] != 'pheme':
            test_f1, test_macro_f1, test_precision, test_recall, test_accuracy, test_loss = self.eval_lr(test=True)
        else:
            fold_preds, fold_labels, test_f1, test_precision, test_recall, test_accuracy, test_loss = self.eval_lr(test=True)
            # test_f1, test_f1_neg, test_macro_f1, test_precision, test_recall, test_accuracy, test_loss = self.eval_lr(test=True)
            
        if config['data_name'] != 'pheme':
            print_test_stats(test_accuracy, test_precision, test_recall, test_f1, test_macro_f1, self.best_val_acc, self.best_val_precision, self.best_val_recall, self.best_val_f1)
            return test_f1, test_macro_f1, test_accuracy
        else:
            # print_test_stats_pheme(test_accuracy, test_precision, test_recall, test_f1, self.best_val_acc, self.best_val_precision, self.best_val_recall, self.best_val_f1)
            # print(self.actual_best_f1)
            return test_f1, fold_preds, fold_labels
            # return test_f1, test_macro_f1
                
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required Paths
    parser.add_argument('--data_path', type = str, default = './data/complete_data',
                          help='path to dataset folder that contains the adj and feat matrices, etc')
    parser.add_argument('--model_checkpoint_path', type = str, default = './model_checkpoints_lr',
                          help='Directory for saving trained model checkpoints')
    parser.add_argument("--model_save_name", type=str, default= 'best_model_lr.pt',
                       help = 'saved model name')
    
    #### Training Params ####
    
    # Named params    
    parser.add_argument('--data_name', type = str, default = 'gossipcop',
                          help='dataset name: politifact / gossipcop / pheme / HealthRelease / HealthStory')
    parser.add_argument('--model_name', type = str, default = 'HGCN',
                          help='model name: gcn / graph_sage / graph_conv / gat / rgcn / HGCN')
    parser.add_argument('--mode', type=str, default='gnn+text',
                        help='what features to use for classification: gnn / text / gnn+text')
    parser.add_argument('--loss_func', type = str, default = 'bce',
                        help = 'Loss function to use for optimization: bce / bce_logits / ce')
    parser.add_argument('--scheduler', type = str, default = 'step',
                        help = 'The type of lr scheduler to use anneal learning rate: step/multi_step')
    parser.add_argument('--optimizer', type = str, default = 'SGD',
                        help = 'Optimizer to use for training')
    
    # Dimensions/sizes params   
    parser.add_argument('--batch_size', type = int, default = 16,
                          help='batch size for training"')
        
    # Numerical params
    parser.add_argument('--lr', type = float, default = 1e-3,
                          help='Learning rate for training')
    parser.add_argument('--weight_decay', type = float, default = 2e-3,
                        help = 'weight decay for optimizer')
    parser.add_argument('--momentum', type = float, default = 0.8,
                        help = 'Momentum for optimizer')
    parser.add_argument('--max_epoch', type = int, default = 50,
                        help = 'Max epochs to train for')
    parser.add_argument('--lr_decay_step', type = float, default = 3,
                        help = 'No. of epochs after which learning rate should be decreased')
    parser.add_argument('--lr_decay_factor', type = float, default = 0.8,
                        help = 'Decay the learning rate of the optimizer by this multiplicative amount')
    parser.add_argument('--patience', type = float, default = 10,
                        help = 'Patience no. of epochs for early stopping')
    parser.add_argument('--seed', type=int, default=168,
                        help='set seed for reproducability')
    
    # Options params
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='whether to shuffle batches')
    
    
    
    
    args, unparsed = parser.parse_known_args()
    config = args.__dict__

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device   

    if config['data_name'] == 'pheme':
        config['n_classes'] = 3
    else:
        config['n_classes'] = 1             
   
    # Check all provided paths:    
    model_path = os.path.join(config['model_checkpoint_path'], config['data_name'], config['model_name'])
    
    if not os.path.exists(config['data_path']):
        raise ValueError("[!] ERROR: Dataset path does not exist")
    else:
        print("\nData path checked..")
    if not os.path.exists(model_path):
        print("\nCreating checkpoint path for saved models at:  {}\n".format(model_path))
        os.makedirs(model_path)
    else:
        print("\nModel save path checked..")
    
    
    

    # Print args
    print("\n" + "x"*50 + "\n\nRunning training with the following parameters: \n")
    for key, value in config.items():
        print(key + ' : ' + str(value))
    print("\n" + "x"*50)
    
    # # Prepare dataset and iterators for training
    # train_loader, val_loader, test_loader = prepare_lr_training(config)
    
    if config['data_name'] != 'pheme':
        seeds = [3, 21, 42, 84, 168]
        # seeds = [3]
        f1_list = []
        macro_f1_list = []
        acc_list = []
        for seed in seeds:
            config['seed']= 21
            print("\nseed= ", seed)
            # if config['model_name'] == 'graph_sage':
            #     config['lr'] = 5e-3
            # Prepare dataset and iterators for training
            train_loader, val_loader, test_loader = prepare_lr_training(config, seed)
            lr_model = LR_Learner(config)
            f1, macro_f1, acc = lr_model.train_main()
            f1_list.append(f1)
            macro_f1_list.append(macro_f1)
            acc_list.append(acc)
            # sys.exit()
        
        print(f1_list)
        print("\nmean accuracy= ", sum(acc_list)/len(acc_list))
        print("std accuracy = ", stdev(acc_list))
        print("\nmean f1= ", sum(f1_list)/len(f1_list))
        print("std f1= ", stdev(f1_list))
        print("\nmean macro-f1= ", sum(macro_f1_list)/len(macro_f1_list))
        print("std macro-f1= ", stdev(macro_f1_list))
    
    else:
        seeds = [3, 21, 42, 84, 168]
        f1_fold_list = []
        for fold in [1,3,4,6,8]:
            f1_list = []
            f1_true_list, f1_false_list = [], []
            test_preds, test_labels = [], []
            f1_unverif_list = []
            for seed in seeds:
                config['seed']= 21
                print("\nseed= ", seed)
                # if config['model_name'] == 'graph_sage':
                #     config['lr'] = 5e-3
                # Prepare dataset and iterators for training
                train_loader, val_loader, test_loader = prepare_lr_training(config, seed, fold)
                lr_model = LR_Learner(config)
                f1, p, l = lr_model.train_main()
                f1_list.append(f1[1])
                test_preds.append(p)
                test_labels.append(l)
                # f1_true_list.append(f1[0])
                # f1_false_list.append(f1[2])
                # f1_unverif_list.append(f1[3])
                # sys.exit()
            
            print("\n FOLD {}:".format(fold))
            print("macro-F1 = ", f1_list)
            print("F1 true = ", f1_true_list)
            print("F1 false = ", f1_false_list)
            print("\nmean f1= ", sum(f1_list)/len(f1_list))
            print("std f1= ", stdev(f1_list))
            # print("\nmean f1 true = ", sum(f1_true_list)/len(f1_true_list))
            # print("mean f1 false = ", sum(f1_false_list)/len(f1_false_list))
            # print("mean f1 unverif = ", sum(f1_unverif_list)/len(f1_unverif_list))
            f1_fold_list.append(sum(f1_list)/len(f1_list))
            
            
        print("\n Average across all folds:")
        print(f1_fold_list)
        print("\nmean f1= ", sum(f1_fold_list)/len(f1_fold_list))
        print("std f1= ", stdev(f1_fold_list))
        test_preds = [pred for batch_pred in test_preds for pred in batch_pred]
        test_labels = [pred for batch_pred in test_labels for pred in batch_pred]
        final_test_f1, final_test_recall, final_test_precision, final_test_accuracy = evaluation_measures_pheme(config, np.array(test_preds), np.array(test_labels))
        print("\n\nFinal Scores (overall):\n")
        print("F1 =", final_test_f1)
        print("Recall = ", final_test_recall)
        print("precision = ", final_test_precision)
        print("Accuracy = ", final_test_accuracy)
        # seeds = [3,21,42,84,168]
        # # seeds = [21,42]
        # f1_fold_list = []
        # macro_f1_fold_list = []
        # for fold in [1,3,4,6,8]:
        #     f1_list = []
        #     macro_f1_list = []
        #     for seed in seeds:
        #         config['seed']= 21
        #         print("\nseed= ", seed)
        #         # if config['model_name'] == 'graph_sage':
        #         #     config['lr'] = 5e-3
        #         # Prepare dataset and iterators for training
        #         train_loader, val_loader, test_loader = prepare_lr_training(config, seed, fold)
        #         lr_model = LR_Learner(config)
        #         f1, macro_f1 = lr_model.train_main()
        #         # f1_list.append(f1[1])
        #         f1_list.append(f1)
        #         macro_f1_list.append(macro_f1)
            
        #     print("\n FOLD {}:".format(fold))
        #     print(f1_list)
        #     print(macro_f1_list)
        #     print("\nmean f1= ", sum(f1_list)/len(f1_list))
        #     print("std f1= ", stdev(f1_list))
        #     print("mean macro_f1= ", sum(macro_f1_list)/len(macro_f1_list))
        #     print("std macro_f1= ", stdev(macro_f1_list))
        #     f1_fold_list.append(sum(f1_list)/len(f1_list))
        #     macro_f1_fold_list.append(sum(macro_f1_list)/len(macro_f1_list))
            
        # print("\n Average across all folds:")
        # print(f1_fold_list)
        # print(macro_f1_fold_list)
        # print("\nmean f1= ", sum(f1_fold_list)/len(f1_fold_list))
        # print("std f1= ", stdev(f1_fold_list))
        # print("mean macro_f1= ", sum(macro_f1_fold_list)/len(macro_f1_fold_list))
        # print("std macro_f1= ", stdev(macro_f1_fold_list))
        
        
    
    # try:
    #     lr_model = LR_Learner(config)
    #     lr_model.train_main()
    # except KeyboardInterrupt:
    #     print("Keyboard interrupt by user detected...\nClosing the tensorboard writer!")
    #     print("Best val f1 = ", lr_model.best_val_f1)