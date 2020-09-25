import argparse, time, datetime, shutil
import sys, os, glob, json, re
import warnings
warnings.filterwarnings("ignore")

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

import nltk
nltk.download('punkt')
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from transformers import get_linear_schedule_with_warmup

from models.model import Document_Classifier
from models.transformer_model import *
from utils.utils import *
from utils.data_utils_gnn import *
from utils.data_utils_txt import *
from utils.data_utils_hygnn import *
from caching_funcs.cache_text import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Doc_Encoder_Main():
    def __init__(self, config):
        self.best_val_acc, self.best_val_f1, self.best_val_recall, self.best_val_precision = 0, 0, 0, 0
        self.preds_list, self.labels_list = [] , []
        self.train_f1, self.train_precision, self.train_recall, self.train_accuracy = 0,0,0,0
        self.loss_list = []
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
        
        # Initialize the model, optimizer and loss function
        self.init_training_params()        
        
        
    def init_training_params(self):
        if self.embed_name == 'glove':
            self.model = Document_Classifier(self.config, pre_trained_embeds = self.config['TEXT'].vocab.vectors).to(device)
        else:
            self.model = Document_Classifier(self.config).to(device)
            
        if self.config['parallel_computing']:
            self.model = nn.DataParallel(self.model)  
            
        if self.config['optimizer'] == 'Adam':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.config['lr'], weight_decay = self.config['weight_decay'])
        elif self.config['optimizer'] == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.config['lr'], momentum = self.config['momentum'], weight_decay = self.config['weight_decay'])
        
        if self.config['loss_func'] == 'bce_logits':
            self.criterion = nn.BCEWithLogitsLoss(pos_weight= torch.tensor([self.config['pos_wt']]).to(device))
        else:
            self.criterion = nn.BCELoss()
        
        
        if self.config['scheduler'] == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config['lr_decay_step'], gamma = self.config['lr_decay_factor'])
        elif self.config['scheduler'] == 'multi_step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones= [2,5,10,15,20,25,35,45], gamma = self.config['lr_decay_factor'])
        elif self.config['scheduler'] == 'warmup':
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, self.config['warmup_steps'], self.config['train_steps'])
        
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



    def calculate_loss(self, preds, batch_label, grad_step=False):
        if self.config['loss_func'] == 'bce':
            preds = F.sigmoid(preds)        
        preds = preds.squeeze(1).to(device) if self.config['loss_func'] == 'bce_logits' else preds.to(device)
        loss = self.criterion(preds,  batch_label.float().squeeze(1).to(device))

        if grad_step:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
        
        if self.config['loss_func'] == 'bce':
            preds = (preds>0.5).type(torch.FloatTensor)
        elif self.config['loss_func'] == 'ce':
            preds = F.softmax(preds, dim=1)
            preds = torch.argmax(preds, dim=1)
        elif self.config['loss_func'] == 'bce_logits': 
            preds = F.sigmoid(preds)
            preds = (preds>0.5).type(torch.FloatTensor).squeeze(1)

        self.preds_list.append(preds.cpu().detach().numpy())
        self.labels_list.append(batch_label.cpu().detach().numpy())
        self.loss_list.append(loss.detach().item())


    
    def eval_model(self, test = False):
        self.model.eval()
        self.preds_list, self.labels_list = [], []
        self.loss_list = []        
    
        with torch.no_grad():
            if self.config['embed_name'] == 'elmo':
                data = self.config['val_data'] if not test else self.config['test_data']
                labels = self.config['val_labels'] if not test else self.config['test_labels']
                rand_idxs = np.random.permutation(len(data))
                data_shuffle = [data[i] for i in rand_idxs]
                label_shuffle = [labels[i] for i in rand_idxs]
                total_iters = int(np.ceil(len(labels)/self.config['batch_size']))
                for iters in range(total_iters):
                    batch_ids, batch_label, sen_lens = get_elmo_batches(self.config, total_iters, iters, data_shuffle, label_shuffle)  
                    preds = self.model(batch_ids.to(device), sen_lens)
                    self.calculate_loss(preds, batch_label)
            else:
                batch_loader = self.config['val_loader'] if not test else self.config['test_loader']
                for iters, batch in enumerate(batch_loader): 
                    if self.config['embed_name'] == 'glove':
                        preds = self.model(inp=batch.text[0].to(device), sent_lens=batch.text[1].to(device))
                        batch_label = batch.label
                    elif self.config['embed_name'] in ['dbert', 'roberta']:
                        preds = self.model(inp= batch['input_ids'], attn_mask = batch['attention_mask'])
                        batch_label = batch['labels'].to(device)
                    self.calculate_loss(preds, batch_label)

            self.preds_list = [pred for batch_pred in self.preds_list for pred in batch_pred]
            self.labels_list = [label for batch_labels in self.labels_list for label in batch_labels]
            
            eval_loss = sum(self.loss_list)/len(self.loss_list)
            eval_f1, eval_macro_f1, eval_recall, eval_precision, eval_accuracy = evaluation_measures(self.config, np.array(self.preds_list), np.array(self.labels_list))
                
        return eval_f1, eval_macro_f1, eval_precision, eval_recall, eval_accuracy, eval_loss
            

    def save_model(self):
        torch.save({
                'epoch': self.epoch,
                'best_val_f1' : self.best_val_f1,
                'model_state_dict': self.model.state_dict(),
                # 'model_classif_state_dict': model.classifier.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, os.path.join(self.model_file))  

    def load_model(self):
        checkpoint = torch.load(self.model_file)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                   
    def generate_summary(self, preds, labels):
        target_names = ['False', 'True', 'Unverified']
        print(classification_report(labels, preds, target_names=target_names))
        return None
    
    

    def check_early_stopping(self):
        self.this_metric = self.eval_f1 if self.config['optimze_for'] == 'f1' else self.eval_loss
        self.current_best = self.best_val_f1 if self.config['optimze_for'] == 'f1' else self.best_val_loss

        new_best = self.this_metric > self.current_best if self.config['optimze_for'] == 'f1' else self.this_metric < self.current_best
        if new_best:
            print("New High Score! Saving model...")
            self.best_val_f1 = self.eval_f1
            self.best_val_loss = self.eval_loss
            self.best_val_acc = self.eval_accuracy
            self.best_val_recall = self.eval_recall
            self.best_val_precision = self.eval_precision
            self.save_model()

        self.scheduler.step()
            
        ### Stopping Criteria based on patience ###        
        diff = self.this_metric - self.current_best if self.config['optimze_for'] == 'f1' else self.current_best - self.this_metric
        if diff < 1e-3:
            self.not_improved+=1
            if self.not_improved >= self.config['patience']:
                self.terminate_training= True
        else:
            self.not_improved = 0
        print("current patience: ", self.not_improved)




    def train_epoch_step(self):
        self.model.train()
        lr = self.scheduler.get_lr()
        self.total_iters += self.iters
        self.preds_list = [pred for batch_pred in self.preds_list for pred in batch_pred]
        self.labels_list = [label for batch_labels in self.labels_list for label in batch_labels]
        
        # Evaluate on train set
        self.train_f1, self.train_f1_macro, self.train_recall, self.train_precision, self.train_accuracy = evaluation_measures(self.config, np.array(self.preds_list), np.array(self.labels_list))  
        log_tensorboard(self.config, self.config['writer'], self.model, self.epoch, self.iters, self.total_iters, self.train_loss, self.train_f1, self.train_precision, self.train_recall, self.train_accuracy, lr[0], self.threshold, loss_only=False, val=False)
        
        # Evaluate on dev set
        self.eval_f1, self.eval_f1_macro, self.eval_precision, self.eval_recall, self.eval_accuracy, self.eval_loss = self.eval_model()

        # print stats
        print_stats(self.config, self.epoch, self.loss_list, self.train_accuracy, self.train_f1, self.train_f1_macro, self.train_precision, self.train_recall,
                    self.eval_loss, self.eval_accuracy, self.eval_f1, self.eval_f1_macro, self.eval_precision, self.eval_recall, self.start, lr[0])
        
        # log validation stats in tensorboard
        log_tensorboard(self.config, self.config['writer'], self.model, self.epoch, self.iters, self.total_iters, self.eval_loss, self.eval_f1, self.eval_precision, self.eval_recall, self.eval_accuracy, lr[0], self.threshold, loss_only = False, val=True)

        # Check for early stopping criteria
        self.check_early_stopping()
                
        self.preds_list = []
        self.labels_list = []
        self.loss_list = []
    



    def end_training(self):
        # Termination message
        if self.terminate_training:
            print("\n" + "-"*100 + "\nTraining terminated early because the Validation loss did not improve for   {}   epochs" .format(self.config['patience']))
        else:
            print("\n" + "-"*100 + "\nMaximum epochs of {} reached. Finished training !!".format(self.config['max_epoch']))
        
        print("\n" + "-"*50 + "\n\t\tEvaluating on test set\n" + "-"*50)
        # self.model_file = os.path.join(self.config['model_checkpoint_path'], self.config['data_name'], self.config['model_name'], self.config['model_save_name'])
        if os.path.isfile(self.model_file):
            self.load_model()            
        else:
            raise ValueError("No Saved model state_dict found for the chosen model...!!! \nAborting evaluation on test set...".format(self.config['model_name']))

        self.test_f1, self.test_f1_macro, self.test_precision, self.test_recall, self.test_accuracy, _ = self.eval_model(test=True)        
        print_test_stats(self.test_accuracy, self.test_precision, self.test_recall, self.test_f1, self.test_f1_macro, self.best_val_acc, self.best_val_precision, self.best_val_recall, self.best_val_f1)       
        self.config['writer'].close()    

    

            
    def train_main(self, cache=False):
        print("\n\n"+ "="*100 + "\n\t\t\t\t\t Training Network\n" + "="*100)
        
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
                    self.preds = self.model(inp=self.batch.text[0].to(device), sent_lens = self.batch.text[1].to(device))
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
                self.train_epoch_step()               
                if self.terminate_training:
                    break
        
        elif self.embed_name in ['dbert', 'xlnet', 'roberta']:
            for self.epoch in range(self.start_epoch, self.config['max_epoch']+1):
                for self.iters, self.batch in enumerate(self.config['train_loader']):
                    self.model.train()
                    self.preds = self.model(inp=self.batch['input_ids'], attn_mask=self.batch['attention_mask'])
                    self.batch_label = self.batch['labels'].to(device)
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
                self.train_epoch_step()
                if self.terminate_training:
                    break
                        
          
        self.end_training()
        if cache:
            Cache_Text_Embeds(self.config, self.model)
        return  self.best_val_f1 , self.best_val_acc, self.best_val_recall, self.best_val_precision, self.test_f1, self.test_f1_macro, self.test_accuracy, self.test_recall, self.test_precision
                    
                
                
    
