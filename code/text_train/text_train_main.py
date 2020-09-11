import argparse, time, datetime, shutil
import sys, os, glob, json, re
import warnings
warnings.filterwarnings("ignore")
# from torchsummary import summary

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

# from tensorboardX import SummaryWriter
# from nltk import word_tokenize
import nltk
nltk.download('punkt')
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report

from models.model import Document_Classifier
from models.transformer_model import *
from utils.utils import *
from caching_funcs.cache_text import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Doc_Encoder_Main():
    def __init__(self, config, train_args=None):
        self.best_val_acc, self.best_val_f1, self.best_val_recall, self.best_val_precision = 0, 0, 0, 0
        self.preds_list, self.labels_list = [] , []
        self.train_f1, self.train_precision, self.train_recall, self.train_accuracy = 0,0,0,0
        self.train_loss = []
        self.prev_val_loss, self.not_improved  = 0, 0
        self.best_val_loss = 1000
        self.total_iters, self.threshold = 0,0
        self.terminate_training = False
        self.model_file = os.path.join(config['model_path'], config['model_save_name'])
        self.start_epoch, self.iters = 1, 0
        self.model_name = config['model_name']
        self.embed_name = config['embed_name']
        self.preds, self.loss = 0, 0
        self.start = time.time()
        self.config = config
        self.train_args = train_args
        
        # Initialize the model, optimizer and loss function
        self.init_training_params()        
        
        
    def init_training_params(self):
        if self.embed_name not in ['bert', 'xlnet', 'roberta']:
            if self.embed_name == 'glove':
                self.model = Document_Classifier(self.config, pre_trained_embeds = self.config['TEXT'].vocab.vectors).to(device)
            elif self.embed_name == 'elmo':
                self.model = Document_Classifier(self.config).to(device)
                
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
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones= [2,5,10,15,20,25,35,45], gamma = self.config['lr_decay_factor'])
        
        return None
        
        
        
    def eval_han_elmo(self, test = False):
        self.model.eval()        
        preds_list, labels_list, eval_loss = [],[], []
        eval_f1_list, eval_recall_list, eval_precision_list, eval_accuracy_list = [], [], [], []
        batch_loader = self.config['val_loader'] if not test else self.config['test_loader']
        
        with torch.no_grad():
            han_iterator = enumerate(batch_loader)
            batch_counter = 0            
            while True:
                try:
                    step, han_data = next(han_iterator)
                except StopIteration:
                    break                
                han_batch = HAN_Batch(self.config['han_max_batch_size'])
                han_batch.add_data(han_data[0], han_data[1].item(), han_data[2].item(), han_data[3])
                if batch_counter > 5:
                    break
                while not han_batch.is_full():
                    try:
                        step, han_data = next(han_iterator)
                    except StopIteration:
                        break
                    han_batch.add_data(han_data[0], han_data[1].item(), han_data[2].item(), han_data[3])
                
                if not self.config['parallel_computing']:
                    han_data = han_batch.pad_and_sort_batch()
                else:
                    han_data = han_batch.just_pad_batch()
                
                batch_ids = han_data[0].long().to(device)
                batch_label = han_data[1].to(device)
                num_of_sents_in_doc = han_data[2].to(device)
                num_tokens_per_sent = han_data[3].to(device)
                batch_counter+= len(num_of_sents_in_doc)
                max_num_sent = batch_ids.shape[1]
                if not self.config['parallel_computing']:
                    # recover_idxs = han_data[4].to(device)
                    preds = self.model(inp = batch_ids, sent_lens = num_tokens_per_sent, doc_lens= num_of_sents_in_doc, arg=max_num_sent)
                else:
                    preds = self.model(inp = batch_ids, sent_lens = num_tokens_per_sent, doc_lens= num_of_sents_in_doc, arg=max_num_sent)
                    
                loss = self.criterion(preds.to(device),  batch_label.float().unsqueeze(1).to(device))
                eval_loss.append(loss.detach().item())
                preds = F.sigmoid(preds)                    
                preds_list.append(preds.cpu().squeeze(1).detach().numpy())
                labels_list.append(batch_label.cpu().detach().numpy())
                
            preds_list = [pred for batch_pred in preds_list for pred in batch_pred]
            labels_list = [label for batch_labels in labels_list for label in batch_labels]
            
            # HAN has reweighted loss. So need to optimize the threshold for binary classification. Check thresholds from 0.2-0.8 with intervals of 0.1
            if not test:
                thresh_list = np.arange(0.2, 0.8, 0.1)
                for thresh in thresh_list:
                    temp_preds_list = (preds_list>thresh)
                    temp_preds_list = [float(i) for i in temp_preds_list]
                    eval_f1, eval_recall, eval_precision, eval_accuracy = evaluation_measures(self.config, np.asarray(temp_preds_list), np.asarray(labels_list))
                    eval_f1_list.append(eval_f1)
                    eval_recall_list.append(eval_recall)
                    eval_precision_list.append(eval_precision)
                    eval_accuracy_list.append(eval_accuracy)
                    
                # We log the validation stats for the threshold that gives best F1
                best_f1 = max(eval_f1_list)
                best_thresh_idx = eval_f1_list.index(max(eval_f1_list))
                if best_f1 > self.best_val_f1:
                    self.threshold = thresh_list[best_thresh_idx] 
                eval_precision = eval_precision_list[best_thresh_idx]
                eval_recall = eval_recall_list[best_thresh_idx]
                eval_accuracy = eval_accuracy_list[best_thresh_idx]
            else:
                # During test, we use the best threshold
                temp_preds_list = (preds_list>self.threshold).type(torch.FloatTensor).squeeze(1)
                best_f1, eval_recall, eval_precision, eval_accuracy = evaluation_measures(self.config, np.asarray(temp_preds_list), np.asarray(labels_list))
                
            eval_loss = sum(eval_loss)/len(eval_loss) 
        return best_f1, eval_precision, eval_recall, eval_accuracy, eval_loss
                
            
    
    def eval_elmo(self, data, labels):
        self.model.eval()
        preds_list, labels_list, eval_loss = [],[], []
        rand_idxs = np.random.permutation(len(data))
        data_shuffle = [data[i] for i in rand_idxs]
        label_shuffle = [labels[i] for i in rand_idxs]
        
        total_iters = int(np.ceil(len(labels)/self.config['batch_size']))
        with torch.no_grad():
            for iters in range(total_iters):
                batch_ids, batch_label, sen_lens = get_elmo_batches(self.config, total_iters, iters, data_shuffle, label_shuffle)  
                preds = self.model(batch_ids.to(device), sen_lens)
                if self.config['loss_func'] == 'bce':
                    preds = F.sigmoid(preds)
                loss = self.criterion(preds,  batch_label.float().to(device))
                eval_loss.append(loss.detach().item())
                if self.config['loss_func'] == 'bce':
                    preds = (preds>0.5).type(torch.FloatTensor)
                elif self.config['loss_func'] == 'ce':
                    preds = F.softmax(preds, dim=1)
                    preds = torch.argmax(preds, dim=1)
                    
                preds_list.append(preds.cpu().detach().numpy())
                labels_list.append(batch_label.cpu().detach().numpy())
                
            preds_list = [pred for batch_pred in preds_list for pred in batch_pred]
            labels_list = [label for batch_labels in labels_list for label in batch_labels]
            
            
            if self.config['data_name'] != 'pheme':
                eval_f1, eval_f1_macro, eval_recall, eval_precision, eval_accuracy = evaluation_measures(self.config, np.array(preds_list), np.array(labels_list))
            else:
                eval_f1, eval_recall, eval_precision, eval_accuracy = evaluation_measures_pheme(self.config, np.array(preds_list), np.array(labels_list))
                eval_f1_macro = None
            
            eval_loss = sum(eval_loss)/len(eval_loss)
            # if test:
            #     self.generate_summary(np.array(preds_list), np.array(labels_list))
        return eval_f1, eval_f1_macro, eval_precision, eval_recall, eval_accuracy, eval_loss
            
        
  
      
    def eval_glove(self, test = False):
        self.model.eval()
        preds_list, labels_list = [], []
        eval_loss = []
        batch_loader = self.config['dev_loader'] if not test else self.config['test_loader']
    
        with torch.no_grad():
            for iters, batch in enumerate(batch_loader): 
                preds = self.model(batch.text[0].to(device), batch.text[1].to(device))
                if self.config['loss_func'] == 'bce':
                    preds = F.sigmoid(preds)
                loss = self.criterion(preds,  batch.label.float().squeeze(1).to(device))
                eval_loss.append(loss.detach().item())
                if self.config['loss_func'] == 'bce':
                    preds = (preds>0.5).type(torch.FloatTensor)
                elif self.config['loss_func'] == 'ce':
                    preds = F.softmax(preds, dim=1)
                    preds = torch.argmax(preds, dim=1)
                preds_list.append(preds.cpu().detach().numpy())
                labels_list.append(batch.label.cpu().detach().numpy())
            
            preds_list = [pred for batch_pred in preds_list for pred in batch_pred]
            labels_list = [label for batch_labels in labels_list for label in batch_labels]
            
            eval_loss = sum(eval_loss)/len(eval_loss)
            if self.config['data_name'] != 'pheme':
                eval_f1, eval_macro_f1, eval_recall, eval_precision, eval_accuracy = evaluation_measures(self.config, np.array(preds_list), np.array(labels_list))
                
            else:
                eval_macro_f1 = None
                eval_f1, eval_recall, eval_precision, eval_accuracy = evaluation_measures_pheme(self.config, np.array(preds_list), np.array(labels_list))
                # eval_f1, eval_f1_neg, eval_macro_f1, eval_recall, eval_precision, eval_accuracy = evaluation_measures_pheme_filtered(self.config, np.array(preds_list), np.array(labels_list))
                
        return eval_f1, eval_macro_f1, eval_precision, eval_recall, eval_accuracy, eval_loss
            
            
            # if test:
            #     self.generate_summary(np.array(preds_list), np.array(labels_list))
        

                   
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
        
        if self.config['data_name'] != 'pheme':
            self.train_f1, self.train_f1_macro, self.train_recall, self.train_precision, self.train_accuracy = evaluation_measures(self.config, np.array(self.preds_list), np.array(self.labels_list))
        else:
            self.train_f1, self.train_recall, self.train_precision, self.train_accuracy = evaluation_measures_pheme(self.config, np.array(self.preds_list), np.array(self.labels_list))
            # self.train_f1, f1_neg, self.train_f1_macro, self.train_recall, self.train_precision, self.train_accuracy = evaluation_measures_pheme_filtered(self.config, np.array(self.preds_list), np.array(self.labels_list))
            
        log_tensorboard(self.config, self.config['writer'], self.model, self.epoch, self.iters, self.total_iters, self.train_loss, self.train_f1, self.train_precision, self.train_recall, self.train_accuracy, lr[0], self.threshold, loss_only=False, val=False)
        
        # Evaluate on dev set
        if self.embed_name == 'glove':
            self.eval_f1, self.eval_f1_macro, self.eval_precision, self.eval_recall, self.eval_accuracy, self.eval_loss = self.eval_glove()
        elif self.embed_name == 'elmo' and self.model_name != 'han':
            self.eval_f1, self.eval_f1_macro, self.eval_precision, self.eval_recall, self.eval_accuracy, self.eval_loss = self.eval_elmo(self.config['val_data'], self.config['val_labels'])
        elif self.embed_name == 'elmo' and self.model_name == 'han':
            self.eval_f1, self.eval_f1_macro, self.eval_precision, self.eval_recall, self.eval_accuracy, self.eval_loss = self.eval_han_elmo()
               
        # print stats
        if self.config['data_name'] != 'pheme':
            print_stats(self.config, self.epoch, self.train_loss, self.train_accuracy, self.train_f1, self.train_f1_macro, self.train_precision, self.train_recall,
                        self.eval_loss, self.eval_accuracy, self.eval_f1, self.eval_f1_macro, self.eval_precision, self.eval_recall, self.start, lr[0])
        else:
            print_stats_pheme(self.config, self.epoch, self.train_loss, self.train_accuracy, self.train_f1, self.train_precision, self.train_recall,
                        self.eval_loss, self.eval_accuracy, self.eval_f1, self.eval_precision, self.eval_recall, self.start, lr[0])
        
        # log validation stats in tensorboard
        log_tensorboard(self.config, self.config['writer'], self.model, self.epoch, self.iters, self.total_iters, self.eval_loss, self.eval_f1, self.eval_precision, self.eval_recall, self.eval_accuracy, lr[0], self.threshold, loss_only = False, val=True)
               
        # # Save model checkpoints for best model
        # if self.eval_loss < self.best_val_loss:
        #     print("New High Score! Saving model...")
        #     self.best_val_f1 = self.eval_f1
        #     self.best_val_f1_macro = self.eval_f1_macro
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
        #     }, os.path.join(self.config['model_checkpoint_path'], self.config['data_name'], self.config['embed_name'], self.config['model_name'], self.config['model_save_name']))
        
        # use macroF1 for dataset with more than 2 classes else use F1 of target class
        # this_f1 = self.eval_f1[1] if self.config['data_name']=='pheme' else self.eval_f1 
        # current_best = self.best_val_f1[1] if (not isinstance(self.best_val_f1, int) and self.config['data_name']=='pheme') else self.best_val_f1
        
        this_f1 = self.eval_f1 
        current_best = self.best_val_f1
        
        if this_f1 > current_best:
            print("New High Score! Saving model...")
            # self.best_val_f1 = self.eval_f1
            self.best_val_f1 = this_f1
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
            }, os.path.join(self.model_file))
            
        self.scheduler.step()
            
        ### Stopping Criteria based on patience ###
        # current_best = self.best_val_f1[1] if (not isinstance(self.best_val_f1, int) and self.config['data_name']=='pheme') else self.best_val_f1
        this_f1 = self.eval_f1 
        current_best = self.best_val_f1
        
        if this_f1 - current_best!=0 and this_f1 - current_best < 1e-3:
            self.not_improved+=1
            print(self.not_improved)
            if self.not_improved >= self.config['patience']:
                self.terminate_training= True
        else:
            self.not_improved = 0
        
        if this_f1 - current_best > 1e-3:
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
        if self.embed_name == 'glove':
            self.loss = self.criterion(self.preds,  self.batch.label.float().squeeze(1).to(device)) # .long() for pheme 3-class
        else:
            self.loss = self.criterion(self.preds.to(device),  self.batch_label.float().to(device))

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
            self.preds = (self.preds>self.threshold).type(torch.FloatTensor).squeeze(1)
            
        if self.embed_name == 'glove':
            self.preds_list.append(self.preds.cpu().detach().numpy())
            self.labels_list.append(self.batch.label.cpu().detach().numpy())
        else:
            self.preds_list.append(self.preds.cpu().detach().numpy())
            self.labels_list.append(self.batch_label.cpu().detach().numpy())

        self.train_loss.append(self.loss.detach().item())
        
        if self.iters%self.config['log_every'] == 0:
            # Loss only
            log_tensorboard(self.config, self.config['writer'], self.model, self.epoch, self.iters, self.total_iters, self.train_loss, self.train_f1, self.train_precision, self.train_recall, self.train_accuracy, loss_only=True, val=False)
            
            
            
            
    def train_main(self, cache=False):
        print("\n\n"+ "="*100 + "\n\t\t\t\t\t Training Network\n" + "="*100)

        # Seeds for reproduceable runs
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        random.seed(self.config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Load the checkpoint to resume training if found
        
        # if os.path.isfile(self.model_file):
        #     checkpoint = torch.load(self.model_file)
        #     self.best_val_f1 = checkpoint['best_val']
        #     self.model.load_state_dict(checkpoint['model_state_dict'])
        #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #     self.start_epoch = checkpoint['epoch'] + 1
        #     self.threshold = checkpoint['threshold']
        #     print("\nResuming training from epoch {} with loaded model and optimizer...\n".format(self.start_epoch))
        #     print("Using the model defined below: \n\n")
        #     print(self.model)
        # else:
        #     print("\nNo Checkpoints found for the chosen model to reusme training... \nTraining the  ''{}''  model from scratch...\n".format(self.config['model_name']))
        #     print("Using the model defined below: \n\n")
        #     print(self.model)
        # print(self.model)
        self.start = time.time()
        print("\nBeginning training at:  {} \n".format(datetime.datetime.now()))
        
        if self.model_name != 'han' and self.embed_name == 'glove':
            # for self.epoch in range(1):
            for self.epoch in range(self.start_epoch, self.config['max_epoch']+1):        
                for self.iters, self.batch in enumerate(self.config['train_loader']):
                    self.model.train()
                    self.preds = self.model(self.batch.text[0].to(device), self.batch.text[1].to(device))
                    self.train_iters_step()
                self.train_epoch_step()
                
                if self.terminate_training:
                    break
                
        elif self.model_name != 'han' and self.embed_name == 'elmo':
            # for self.epoch in range(1):
            for self.epoch in range(self.start_epoch, self.config['max_epoch']+1):
                rand_idxs = np.random.permutation(len(self.config['train_data']))
                train_data_shuffle = [self.config['train_data'][i] for i in rand_idxs]
                train_label_shuffle = [self.config['train_labels'][i] for i in rand_idxs]
                
                self.max_iters = int(np.ceil(len(train_label_shuffle)/self.config['batch_size']))
                for self.iters in range(self.max_iters):
                    self.model.train()                    
                    self.batch_ids, self.batch_label, self.sen_lens = get_elmo_batches(self.config, self.max_iters, self.iters, train_data_shuffle, train_label_shuffle)
                    self.preds = self.model(self.batch_ids.to(device), self.sen_lens)
                    self.train_iters_step()
                self.train_epoch_step()               
                if self.terminate_training:
                    break
        
        elif self.model_name == 'han' and self.embed_name == 'elmo':
            # for self.epoch in range(1):
            for self.epoch in range(self.start_epoch, self.config['max_epoch']+1):
                # with torch.autograd.set_detect_anomaly(True):
                self.model.train()
                self.total_len_train = len(self.config['train_loader'])
                han_iterator = enumerate(self.config['train_loader'])
                self.batch_counter = 0
                self.iters=0               
                while True:
                    self.iters+=1
                    try:
                        step, han_data = next(han_iterator)
                    except StopIteration:
                        break                    
                    han_batch = HAN_Batch(self.config['han_max_batch_size'])
                    han_batch.add_data(han_data[0], han_data[1].item(), han_data[2].item(), han_data[3])
                    if self.batch_counter > 10:
                        break
                    while not han_batch.is_full():
                        try:
                            step, han_data = next(han_iterator)
                        except StopIteration:
                            break
                        han_batch.add_data(han_data[0], han_data[1].item(), han_data[2].item(), han_data[3])
                        
                    if not self.config['parallel_computing']:
                        han_data = han_batch.pad_and_sort_batch()
                    else:
                        han_data = han_batch.just_pad_batch()

                    self.batch_ids = han_data[0].long().to(device)
                    self.batch_label = han_data[1].to(device)
                    self.num_of_sents_in_doc = han_data[2].to(device)
                    self.num_tokens_per_sent = han_data[3].to(device)
                    self.batch_counter+= len(self.num_of_sents_in_doc)
                    max_num_sent = self.batch_ids.shape[1]
                    
                    if not self.config['parallel_computing']:
                        # self.recover_idxs = han_data[4].to(device)
                        self.preds = self.model(inp = self.batch_ids, sent_lens = self.num_tokens_per_sent, doc_lens= self.num_of_sents_in_doc, arg=max_num_sent)
                    else:
                        self.preds = self.model(inp = self.batch_ids, sent_lens = self.num_tokens_per_sent, doc_lens= self.num_of_sents_in_doc, arg=max_num_sent)
                    self.train_iters_step()
                self.train_epoch_step()
                if self.terminate_training:
                    break
            
            
        elif self.embed_name in ['bert', 'xlnet', 'roberta']:
            
            if self.config['data_name'] != 'pheme':
                num_labels = 2
            else:
                num_labels = 3
                
            self.model = TRANSFORMER(self.embed_name, self.model_name, args = self.train_args, num_labels=num_labels, use_cuda = self.config['use_cuda'], classif_type = self.config['classifier'], \
                                     extract_embeddings = self.config['extract_embeddings'], sliding_window = self.config['sliding_window'], hidden_dropout_prob= self.config['hidden_dropout_prob'], attention_probs_dropout_prob= self.config['attention_probs_dropout_prob'])
            
            # Fine-tune BERT on the train-set
            self.model.train_model(self.config['train_df'], eval_df=self.config['val_df'], writer= self.config['writer'], freeze= self.config['freeze'])
            
            # Load the best model
            print("\n" + "Loading best model for eval \n" + "-"*35)
            self.model = TRANSFORMER(self.embed_name, self.train_args['output_dir'], args = self.train_args, num_labels=num_labels, use_cuda = self.config['use_cuda'], classif_type = self.config['classifier'], \
                                     extract_embeddings = self.config['extract_embeddings'], sliding_window = self.config['sliding_window'], hidden_dropout_prob= self.config['hidden_dropout_prob'], attention_probs_dropout_prob= self.config['attention_probs_dropout_prob'])
                
            
            if not cache:
                # # Get evaluation results for each set
                # train_result, _, _, train_stats, train_embeds, train_labels = self.model.eval_model(train_df, test=True)
                val_result, _, _, val_stats, val_embeds, val_labels = self.model.eval_model(self.config['val_df'], test=True)
                test_result, _, _, test_stats, test_embeds, test_labels = self.model.eval_model(self.config['test_df'], test=True)
                print_transformer_results(self.config, val_stats, test_stats, val_result, test_result)
                
            
            if cache and self.config['data_name'] == 'pheme':
                # # Get evaluation results for each set
                train_result, _, _, train_stats, train_embeds, train_labels = self.model.eval_model(self.config['train_df'], test=True)
                val_result, _, _, val_stats, val_embeds, val_labels = self.model.eval_model(self.config['val_df'], test=True)
                test_result, _, _, test_stats, test_embeds, test_labels = self.model.eval_model(self.config['test_df'], test=True)
                
                base_dir = os.path.join('data', 'complete_data', 'pheme_cv')
                
                doc_embed_file = os.path.join(base_dir, 'fold_{}'.format(self.config['fold']), 'doc_embeds_roberta_lr_filtered_{}_train.pt'.format(self.train_args['manual_seed']))
                print("\nSaving train docs embeddings in : ", doc_embed_file)
                print(train_embeds.shape)
                torch.save(train_embeds, doc_embed_file)

                
                doc_embed_file = os.path.join(base_dir, 'fold_{}'.format(self.config['fold']), 'doc_embeds_roberta_lr_filtered_{}_test.pt'.format(self.train_args['manual_seed']))
                print("\nSaving test docs embeddings in : ", doc_embed_file)
                print(test_embeds.shape)
                torch.save(test_embeds, doc_embed_file)
                
                labels_dict = {}
                labels_dict['train'] = list(map(int, train_labels.cpu().numpy()))
                # labels_dict['val'] = list(map(int, val_labels.cpu().numpy()))
                labels_dict['test'] = list(map(int, test_labels.cpu().numpy()))
                labels_file = os.path.join(base_dir, 'fold_{}'.format(self.config['fold']), 'roberta_labels_filtered_{}.json'.format(self.train_args['manual_seed']))
                with open(labels_file, 'w+') as json_file:
                        json.dump(labels_dict, json_file)
                
                # Printing results               
                print_transformer_results(self.config, val_stats, test_stats, val_result, test_result)
                Cache_Text_Embeds(self.config, self.model)
                        
                        
            elif cache and self.config['data_name'] != 'pheme':
                # # Get evaluation results for each set
                train_result, _, _, train_stats, train_embeds, train_labels = self.model.eval_model(self.config['train_df'], test=True)
                val_result, _, _, val_stats, val_embeds, val_labels = self.model.eval_model(self.config['val_df'], test=True)
                test_result, _, _, test_stats, test_embeds, test_labels = self.model.eval_model(self.config['test_df'], test=True)
                
                base_dir = os.path.join('data', 'complete_data', self.config['data_name'])
                
                doc_embed_file = os.path.join(base_dir, 'cached_embeds', 'doc_embeds_roberta_lr_{}_train.pt'.format(self.train_args['manual_seed']))
                print("\nSaving train docs embeddings in : ", doc_embed_file)
                print(train_embeds.shape)
                torch.save(train_embeds, doc_embed_file)
                
                doc_embed_file = os.path.join(base_dir, 'cached_embeds', 'doc_embeds_roberta_lr_{}_val.pt'.format(self.train_args['manual_seed']))
                print("\nSaving val docs embeddings in : ", doc_embed_file)
                print(val_embeds.shape)
                torch.save(val_embeds, doc_embed_file)
                
                doc_embed_file = os.path.join(base_dir, 'cached_embeds', 'doc_embeds_roberta_lr_{}_test.pt'.format(self.train_args['manual_seed']))
                print("\nSaving test docs embeddings in : ", doc_embed_file)
                print(test_embeds.shape)
                torch.save(test_embeds, doc_embed_file)
                
                labels_dict = {}
                labels_dict['train'] = list(map(int, train_labels.cpu().numpy()))
                labels_dict['val'] = list(map(int, val_labels.cpu().numpy()))
                labels_dict['test'] = list(map(int, test_labels.cpu().numpy()))
                labels_file = os.path.join(base_dir, 'cached_embeds', 'roberta_labels_{}.json'.format(self.train_args['manual_seed']))
                with open(labels_file, 'w+') as json_file:
                        json.dump(labels_dict, json_file)
                
                # Printing results               
                print_transformer_results(self.config, val_stats, test_stats, val_result, test_result)
                Cache_Text_Embeds(self.config, self.model)

            
            self.config['writer'].close()
            return val_stats, test_stats, val_result['mcc'], test_result['mcc']
            
            
        if self.embed_name not in ['bert', 'xlnet', 'roberta']:
            # Termination message
            if self.terminate_training:
                print("\n" + "-"*100 + "\nTraining terminated early because the Validation loss did not improve for   {}   epochs" .format(self.config['patience']))
            else:
                print("\n" + "-"*100 + "\nMaximum epochs of {} reached. Finished training !!".format(self.config['max_epoch']))
            
            print("\n" + "-"*50 + "\n\t\tEvaluating on test set\n" + "-"*50)
            # self.model_file = os.path.join(self.config['model_checkpoint_path'], self.config['data_name'], self.config['model_name'], self.config['model_save_name'])
            if os.path.isfile(self.model_file):
                checkpoint = torch.load(self.model_file)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                raise ValueError("No Saved model state_dict found for the chosen model...!!! \nAborting evaluation on test set...".format(self.config['model_name']))
            if self.embed_name == 'glove':
                test_f1, test_f1_macro, test_precision, test_recall, test_accuracy, test_loss = self.eval_glove(test=True)
            elif self.model_name != 'han' and self.embed_name == 'elmo':
                test_f1, test_f1_macro, test_precision, test_recall, test_accuracy, test_loss = self.eval_elmo(test_data, test_label)
            elif self.model_name == 'han' and self.embed_name == 'elmo':
                test_f1, test_f1_macro, test_precision, test_recall, test_accuracy, test_loss = self.eval_han_elmo(test= True)
            if self.config['data_name'] != 'pheme':
                print_test_stats(test_accuracy, test_precision, test_recall, test_f1, test_f1_macro, self.best_val_acc, self.best_val_precision, self.best_val_recall, self.best_val_f1)
            else:
                print_test_stats_pheme(test_accuracy, test_precision, test_recall, test_f1, self.best_val_acc, self.best_val_precision, self.best_val_recall, self.best_val_f1)
            
            if cache:
                Cache_Text_Embeds(self.config, self.model)
            self.config['writer'].close()
            return  self.best_val_f1 , self.best_val_acc, self.best_val_recall, self.best_val_precision, test_f1, test_f1_macro, test_accuracy, test_recall, test_precision
                    
                
                
    
