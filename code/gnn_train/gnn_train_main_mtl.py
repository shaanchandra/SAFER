import torch
import torch.nn.functional as F
import argparse, time, datetime, shutil,random
import sys, os, glob, json
import warnings
warnings.filterwarnings("ignore")
# from torchsummary import summary

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix, save_npz, load_npz
from sklearn.metrics import accuracy_score, classification_report

from torch.utils.tensorboard import SummaryWriter

# from tensorboardX import SummaryWriter
# from nltk import word_tokenize
from torch.autograd import Variable
from torchtext.data import Field, BucketIterator
from torchtext import datasets
import torch.nn as nn
from sklearn.metrics import accuracy_score

from models.gnn_model import Graph_Net, Graph_Net_MTL, Relational_GNN
from models.transformer_model import *
from utils.utils import *
from caching_funcs.cache_gnn import *
from utils.gnn_explainer import GNNExplainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        


class Graph_Net_MTL_Main():
    def __init__(self, config):
        self.config = config
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
        self.model_file = os.path.join(self.config['model_checkpoint_path'], self.config['data_name'], self.config['model_name'], self.config['model_save_name'])
        self.config['model_file'] = self.model_file
        self.start_epoch, self.iters = 0, 0
        self.model_name = self.config['model_name']
        self.preds, self.loss = 0, 0
        self.start = time.time()
        
        # Initialize the model, optimizer and loss function
        self.init_training_params()        
        
        
    def init_training_params(self):
        self.model = Graph_Net_MTL(self.config).to(device) if not self.config['model_name'].startswith('r') else Relational_GNN(self.config).to(device)
         
        if self.config['parallel_computing']:
            self.model = nn.DataParallel(self.model)  
            
        if self.config['optimizer'] == 'Adam':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.config['lr'], weight_decay = self.config['weight_decay'])
        elif self.config['optimizer'] == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.config['lr'], momentum = self.config['momentum'], weight_decay = self.config['weight_decay'])
          
        if self.config['data_name'] != 'pheme':
            if self.config['loss_func'] == 'bce_logits':
                self.criterion = nn.BCEWithLogitsLoss(pos_weight= torch.tensor([self.config['pos_wt']]).to(device))
            else:
                self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()            
        
        if self.config['scheduler'] == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config['lr_decay_step'], gamma = self.config['lr_decay_factor'])
        elif self.config['scheduler'] == 'multi_step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones= [5,10,15,20,30,40,55], gamma = self.config['lr_decay_factor'])
            
            
    
    
    def eval_gcn(self, test = True):
        self.model.eval()
        preds_list, labels_list = [], []
        eval_loss = []
    
        with torch.no_grad():
            if not self.config['full_graph']:
                loader = self.config['loader_veracity'] if self.epoch%2 == 0 else self.config['loader_stance']
                # loader = self.config['loader_stance']
                if self.config['fold'] == 5:
                    loader = self.config['loader_veracity']
                for iters, eval_data in enumerate(loader): 
                    if self.config['model_name'] in ['rgcn', 'rgat', 'rsage']:
                        preds, node_drop_mask, _ = self.model(eval_data.x.to(device), eval_data.edge_index.to(device), eval_data.edge_attr.to(device), self.epoch)
                    else:
                        preds, node_drop_mask, _ = self.model(eval_data.x.to(device), eval_data.edge_index.to(device), epoch=self.epoch)
                        
                    if self.config['loss_func'] == 'bce':
                        preds = F.sigmoid(preds)
                    
                    if self.config['fold'] != 5:
                        preds = preds[eval_data.test_mask_veracity==1] if self.epoch%2==0 else preds[eval_data.test_mask_stance==1]
                        labels = eval_data.y[eval_data.test_mask_veracity==1].long() if self.epoch%2==0 else eval_data.y[eval_data.test_mask_stance==1].long()
                    else:
                        preds = preds[eval_data.test_mask_veracity==1] 
                        labels = eval_data.y[eval_data.test_mask_veracity==1].long()
                    # preds = preds[eval_data.test_mask_stance==1]
                    # labels = eval_data.y[eval_data.test_mask_stance==1].long()
                    
                    # labels = data.y * node_drop_mask.squeeze()
                    # labels = labels[data.test_mask==1]
                    loss = self.criterion(preds.to(device), labels.to(device))
                    eval_loss.append(loss.detach().item())
                    if self.config['loss_func'] == 'ce':
                        preds = F.softmax(preds, dim=1)
                        preds = torch.argmax(preds, dim=1)
                    else:
                        preds = (F.sigmoid(preds)>0.5).type(torch.FloatTensor) if self.config['loss_func'] == 'bce_logits' else (preds>0.5).type(torch.FloatTensor)
                    preds_list.append(preds.cpu().detach().numpy())
                    labels_list.append(labels.cpu().detach().numpy())
            
            else:
                # data_x = data.x * data.representation_mask.unsqueeze(1) if not test else data.x # if not test and self.config['mode'] == 'normal' else data.x
                # data_x = data.x * data.representation_mask.unsqueeze(1) if not test and self.config['mode'] == 'normal' else data.x
                if self.config['model_name'] in ['rgcn', 'rgat', 'rsage']:
                    preds, node_drop_mask, _ = self.model(self.config['data'].x.to(device), self.config['data'].edge_index.to(device), self.config['data'].edge_attr.to(device))
                else:
                    preds, node_drop_mask, _ = self.model(self.config['data'].x.to(device), self.config['data'].edge_index.to(device))
                if self.config['loss_func'] == 'bce':
                    preds = F.sigmoid(preds)
                if test:
                    preds = preds[self.config['data'].test_mask==1]
                    labels = self.config['data'].y[self.config['data'].test_mask==1].long() if self.config['data_name'] == 'pheme' else self.config['data'].y[self.config['data'].test_mask==1].float()
                else:
                    preds = preds[self.config['data'].val_mask==1]
                    labels = self.config['data'].y[self.config['data'].val_mask==1].long() if self.config['data_name'] == 'pheme' else self.config['data'].y[self.config['data'].val_mask==1].float()
                # labels = self.data.y * node_drop_mask.squeeze()
                # labels = labels[self.data.test_mask==1]
                loss = self.criterion(preds.to(device), labels.to(device))
                eval_loss.append(loss.detach().item())
                if self.config['loss_func'] == 'ce':
                    preds = F.softmax(preds, dim=1)
                    preds = torch.argmax(preds, dim=1)
                else:
                    preds = (F.sigmoid(preds)>0.5).type(torch.FloatTensor) if self.config['loss_func'] == 'bce_logits' else (preds>0.5).type(torch.FloatTensor)
                preds_list.append(preds.cpu().detach().numpy())
                labels_list.append(labels.cpu().detach().numpy())
                
            preds_list = [pred for batch_pred in preds_list for pred in batch_pred]
            labels_list = [label for batch_labels in labels_list for label in batch_labels]
            eval_loss = sum(eval_loss)/len(eval_loss)
            
            eval_macro_f1 = None
            eval_f1, eval_recall, eval_precision, eval_accuracy = evaluation_measures_pheme(self.config, np.array(preds_list), np.array(labels_list))
        return eval_f1, eval_macro_f1, eval_precision, eval_recall, eval_accuracy, eval_loss
        
    
    def generate_summary(self, preds, labels):
        target_names = ['False', 'True', 'Unverified']
        print(classification_report(labels, preds, target_names=target_names))
        return None
    
    
    
    
                


        
    def train_epoch_step(self):
        self.model.train()
        lr = self.scheduler.get_lr()
        self.total_iters += self.iters
        self.preds_list = [pred for batch_pred in self.preds_list for pred in batch_pred]
        self.labels_list = [label for batch_labels in self.labels_list for label in batch_labels]
        
        self.train_f1, self.train_recall, self.train_precision, self.train_accuracy = evaluation_measures_pheme(self.config, np.array(self.preds_list), np.array(self.labels_list))
            
        log_tensorboard(self.config, self.config['writer'], self.model, self.epoch, self.iters, self.total_iters, self.train_loss, self.train_f1, self.train_precision, self.train_recall, self.train_accuracy, lr[0], self.threshold, loss_only=False, val=False)
        
        # Evaluate on dev set
        self.eval_f1, self.eval_macro_f1, self.eval_precision, self.eval_recall, self.eval_accuracy, self.eval_loss = self.eval_gcn()
               
        # print stats
        print_stats_pheme(self.config, self.epoch+1, self.train_loss, self.train_accuracy, self.train_f1, self.train_precision, self.train_recall,
                        self.eval_loss, self.eval_accuracy, self.eval_f1, self.eval_precision, self.eval_recall, self.start, lr[0])
        
        # log validation stats in tensorboard
        log_tensorboard(self.config, self.config['writer'], self.model, self.epoch, self.iters, self.total_iters, self.eval_loss, self.eval_f1, self.eval_precision, self.eval_recall, self.eval_accuracy, lr[0], self.threshold, loss_only = False, val=True)
               
        # # Save model checkpoints for best model
        # if self.eval_loss < self.best_val_loss:
        #     print("New High Score! Saving model...")
        #     self.best_val_f1 = self.eval_f1
        #     self.best_val_acc = self.eval_accuracy
        #     self.best_val_recall = self.eval_recall
        #     self.best_val_precision = self.eval_precision
        #     best_model = self.model.state_dict()
        #     # Save the state and the vocabulary
        #     torch.save({
        #         'epoch': self.epoch,
        #         'best_val_f1' : self.best_val_f1,
        #         'model_state_dict': best_model,
        #         # 'model_classif_state_dict': model.classifier.state_dict(),
        #         'optimizer_state_dict': self.optimizer.state_dict(),
        #     }, os.path.join(self.model_file))
        
        if self.epoch%2==0:
            this_f1 = self.eval_f1[1] if self.config['data_name']=='pheme' else self.eval_f1
            if this_f1 > (self.best_val_f1[1] if not isinstance(self.best_val_f1, int) else self.best_val_f1):
                print("New High Score! Saving model...")
                self.best_val_f1 = self.eval_f1
                # self.best_val_f1 = this_f1
                self.best_val_f1_macro = self.eval_macro_f1
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
                }, os.path.join(self.model_file))
                
            if self.epoch==0:
                print("Saving model !")
                best_model = self.model.state_dict()
                # Save the state and the vocabulary
                torch.save({
                    'epoch': self.epoch,
                    'best_val_f1' : self.best_val_f1,
                    'model_state_dict': best_model,
                    # 'model_classif_state_dict': model.classifier.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, os.path.join(self.model_file))
                
            self.scheduler.step()
        
                
            ### Stopping Criteria based on patience ####
                
            # If validation f1 score does not improve, divide the learning rate by 5 and
            # if learning rate falls below given threshold, then terminate training
            current_best = self.best_val_f1[1] if (not isinstance(self.best_val_f1, int) and self.config['data_name']=='pheme') else self.best_val_f1
            
            if this_f1 - current_best!=0 and this_f1 - current_best < 1e-3:
                self.not_improved+=1
                print(self.not_improved)
                if self.not_improved >= self.config['patience']:
                    self.terminate_training= True
            else:
                self.not_improved = 0
            
            if this_f1 > current_best and this_f1 - current_best > 1e-3:
                self.best_val_f1 = self.eval_f1
                self.not_improved=0        
            
            # if self.best_val_loss - self.eval_loss < 1e-3:
            #     self.not_improved+=1
            #     print(self.not_improved)
            #     if self.not_improved >= self.config['patience']:
            #         self.terminate_training= True
            # else:
            #     self.not_improved = 0
            
            # if self.eval_loss < self.best_val_loss and self.best_val_loss - self.eval_loss > 1e-3:
            #     self.best_val_loss = self.eval_loss
            #     self.not_improved = 0
            
            
        self.preds_list = []
        self.labels_list = []
        
        
        
        
    def train_iters_step(self):
        if self.config['loss_func'] == 'bce':
            self.preds = F.sigmoid(self.preds)
        
        if self.config['fold'] != 5:
            self.preds = self.preds[self.data.train_mask_veracity==1] if self.epoch%2==0 else self.preds[self.data.train_mask_stance==1]
        else:
            self.preds = self.preds[self.data.train_mask_veracity==1]
        # self.preds = self.preds[self.data.train_mask_stance==1]
        # self.batch_labels = self.data.y * self.node_drop_mask.squeeze()
        # self.batch_labels = self.batch_labels[self.data.train_mask==1] 
        
        if self.config['fold'] != 5:
            self.batch_labels = self.data.y[self.data.train_mask_veracity==1].long() if self.epoch%2==0 else self.data.y[self.data.train_mask_stance==1].long()
        else:
            self.batch_labels = self.data.y[self.data.train_mask_veracity==1].long()
            
        # self.batch_labels = self.data.y[self.data.train_mask_stance==1].long()
        self.loss = self.criterion(self.preds.to(device),  self.batch_labels.to(device))
        self.loss = self.loss if self.epoch%2==0 else 0.5*self.loss

        self.optimizer.zero_grad()
        self.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.optimizer.step()
    

        if self.config['loss_func'] == 'ce':
            self.preds = F.softmax(self.preds, dim=1)
            self.preds = torch.argmax(self.preds, dim=1)
        elif self.config['loss_func'] == 'bce':
            self.preds = (self.preds>0.5).type(torch.FloatTensor)
        elif self.config['loss_func'] == 'bce_logits': 
            self.preds = F.sigmoid(self.preds)
            self.preds = (self.preds>0.4).type(torch.FloatTensor).squeeze(1)
            
        self.preds_list.append(self.preds.cpu().detach().numpy())
        self.labels_list.append(self.batch_labels.cpu().detach().numpy())

        self.train_loss.append(self.loss.detach().item())
        
        if self.iters%self.config['log_every'] == 0:
            # Loss only
            log_tensorboard(self.config, self.config['writer'], self.model, self.epoch, self.iters, self.total_iters, self.train_loss, self.train_f1, self.train_precision, self.train_recall, self.train_accuracy, loss_only=True, val=False)
    
    
    
    def train_main(self, cache):
        print("\n\n"+ "="*100 + "\n\t\t\t\t\t Training Network\n" + "="*100)

        # Seeds for reproduceable runs
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        random.seed(self.config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.start = time.time()
        print("\nBeginning training at:  {} \n".format(datetime.datetime.now()))
        
        if not self.config['full_graph']:
            for self.epoch in range(self.start_epoch, self.config['max_epoch']+1):
                loader = self.config['loader_veracity'] if self.epoch%2==0 else self.config['loader_stance']
                # loader = self.config['loader_stance']
                if self.config['fold'] == 5:
                    loader = self.config['loader_veracity']
                for self.iters, self.data in enumerate(loader):
                    self.model.train()
                    if self.config['fold']!=5:
                        data_x = self.data.x * self.data.representation_mask_veracity.unsqueeze(1) if self.epoch%2==0 else self.data.x * self.data.representation_mask_stance.unsqueeze(1) # if self.config['mode'] == 'normal' else self.data.x
                    else:
                        data_x = self.data.x * self.data.representation_mask_veracity.unsqueeze(1)
                    # data_x = self.data.x * self.data.representation_mask_stance.unsqueeze(1)
                    if self.config['model_name'] in ['rgcn', 'rgat', 'rsage']:
                        self.preds, self.node_drop_mask, _ = self.model(data_x.to(device), self.data.edge_index.to(device), self.data.edge_attr.to(device), self.epoch)
                    else:
                        self.preds, self.node_drop_mask, _ = self.model(data_x.to(device), self.data.edge_index.to(device), epoch=self.epoch)
                    self.train_iters_step()
                self.train_epoch_step()
                
                if self.terminate_training:
                    break       
        else:
            for self.epoch in range(self.start_epoch, self.config['max_epoch']+1):
                self.model.train()
                self.iters = self.epoch
                self.data = self.config['data_veracity'] if self.epoch%2==0 else self.config['data_stance']
                data_x = self.data.x * self.data.representation_mask.unsqueeze(1) # if self.config['mode'] == 'normal' else self.data.x
                if self.config['model_name'] in ['rgcn', 'rgat', 'rsage']:
                    self.preds, self.node_drop_mask, _ = self.model(data_x.to(device), self.data.edge_index.to(device), self.data.edge_attr.to(device), self.epoch)
                else:
                    self.preds, self.node_drop_mask, _ = self.model(data_x.to(device), self.data.edge_index.to(device))
                self.train_iters_step()
                self.train_epoch_step()
                
                if self.terminate_training:
                    break
        
        # Termination message
        if self.terminate_training:
            print("\n" + "-"*100 + "\nTraining terminated early because the Validation loss did not improve for   {}   epochs" .format(self.config['patience']))
        else:
            print("\n" + "-"*100 + "\nMaximum epochs reached. Finished training !!")
        
        if self.config['mode'] == 'normal':
            if self.epoch > self.config['patience']:
                print("\n" + "-"*50 + "\n\t\tEvaluating on test set\n" + "-"*50)
                if os.path.isfile(self.model_file):
                    checkpoint = torch.load(self.model_file)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                else:
                    raise ValueError("No Saved model state_dict found for the chosen model...!!! \nAborting evaluation on test set...".format(self.config['model_name']))
                
                test_f1, test_macro_f1, test_precision, test_recall, test_accuracy, test_loss = self.eval_gcn(test=True)
                if self.config['data_name'] != 'pheme':
                    print_test_stats(test_accuracy, test_precision, test_recall, test_f1, test_macro_f1, self.best_val_acc, self.best_val_precision, self.best_val_recall, self.best_val_f1)
                else:
                    print_test_stats_pheme(test_accuracy, test_precision, test_recall, test_f1, self.best_val_acc, self.best_val_precision, self.best_val_recall, self.best_val_f1)
            else:
                test_f1, test_macro_f1, test_precision, test_recall, test_accuracy = 0,0,0,0,0
                
            #self.predict_and_cache()
            self.config['writer'].close()
            return  self.best_val_f1 , self.best_val_acc, self.best_val_recall, self.best_val_precision, test_f1, test_macro_f1, test_accuracy, test_recall, test_precision
        
        else:
            if cache:
                Cache_GNN_Embeds(self.config, self.model)
            
            # print("\nModel explainer working...")
            # checkpoint = torch.load(self.model_file, map_location=torch.device('cpu'))
            # self.model.load_state_dict(checkpoint['model_state_dict'])
            # explainer = GNNExplainer(self.model, epochs=200)
            # node_idx = 8000
            # node_feat_mask, edge_mask = explainer.explain_node(node_idx, self.config['data_veracity'].x, self.config['data_veracity'].edge_index)
            # ax, G = explainer.visualize_subgraph(node_idx, self.config['data_veracity'].edge_index, edge_mask, y= self.config['data_veracity'].node_type, threshold = None)
            # # plt.savefig(fname = './data/{}_explained_node{}.pdf'.format(self.config['model_name'], str(node_idx)), dpi=25)
            # plt.tight_layout()
            # plt.show()
            return  self.best_val_f1 , self.best_val_acc, self.best_val_recall, self.best_val_precision
        
        
        
        