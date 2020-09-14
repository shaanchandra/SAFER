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
sys.path.append("..")
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report

from models.model import Document_Classifier
from models.transformer_model import *
from utils.utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Cache_Text_Embeds():
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.comp_dir = os.path.join('data', 'complete_data', config['data_name'])
        self.data_dir = os.path.join('FakeHealth')

        
        if config['data_name'] in ['gossipcop', 'politifcat']:
            self.predict_and_cache_fakenews()
        elif config['data_name'] in ['HealthStory', 'HealthRelease']:
            self.predict_and_cache_fakehealth()
                        
                        
                        
                        
        
    def predict_and_cache_fakenews(self):
        print("\n\nCaching FakeNews dataset : ", self.config['data_name'])
        docs_splits_file = os.path.join(self.comp_dir, 'doc_splits_lr.json')
        doc_splits = json.load(open(docs_splits_file, 'r'))
        test_docs = doc_splits['test_docs']
        train_docs = doc_splits['train_docs']
        val_docs = doc_splits['val_docs']
        
        # Creating doc2id dictionary
        print("\nCreating doc2id dictionary..")
        splits = ['fake', 'real']
        doc2id = {}
        for split in splits:
            src_dir = os.path.join('data', 'base_data', self.config['data_name'], split)
            for root, dirs, files in os.walk(src_dir):
                for count,file in enumerate(files):
                    doc= file.split('.')[0]
                    # if str(doc) in test_docs:
                    doc2id[str(doc)] = len(doc2id)
        
        name = 'doc2id_encoder.json' if self.config['embed_name'] == 'roberta' else 'doc2id_cnn_encoder.json'
        doc2id_file = os.path.join(self.comp_dir, name)
        print("doc2id = ", len(doc2id))
        print("Saving in : ", doc2id_file)
        with open(doc2id_file, 'w+') as json_file:
            json.dump(doc2id, json_file)      
                
        if self.config['embed_name'] in ['glove', 'elmo']:
            self.config['batch_size'] = 1
            self.config['train_loader'], self.config['dev_loader'], self.config['test_loader'], self.config['TEXT'], self.config['LABEL'], self.config['train_split'], self.config['val_split'], self.config['test_split'] = prepare_glove_training(self.config)
            
            # print("\nPreparing the GLoVE embeddings..")
            # pre_trained_embeds = self.config['TEXT'].vocab.vectors
            # embedding = nn.Embedding(self.config['vocab_size'], 300)
            # embedding.weight.data.copy_(pre_trained_embeds)
            # embedding.requires_grad = False
            
        
        # iterating over test_docs and saving their representations
        splits = ['train', 'val', 'test']
        for split in splits:
            print("\nObtaining {} doc representations...".format(split))
            if split == 'train':
                split_docs = train_docs
            elif split == 'val':
                split_docs = val_docs
            else:
                split_docs = test_docs
            
            embed_dim = 1024 if self.config['embed_name'] == 'roberta' else 384
            split_doc_cache = torch.zeros(len(doc2id), embed_dim).to(device)
            print("split_doc_cache shape = ", split_doc_cache.shape)
            not_found=0
            with torch.no_grad():
                if self.config['embed_name'] == 'roberta':
                    for count, doc in enumerate(split_docs):
                        real_path = os.path.join('data', 'base_data', self.config['data_name'], 'real', str(doc)+'.json')
                        fake_path = os.path.join('data', 'base_data', self.config['data_name'], 'fake', str(doc)+'.json')
                        doc_file = real_path if os.path.isfile(real_path) else fake_path
                        if os.path.isfile(doc_file):
                            text_list= []
                            text = json.load(open(doc_file, 'r') )
                            text = text['text'].replace('\n', ' ')
                            text = text.replace('\t', ' ')
                            text_list.append(str(text))
                            _, _, doc_embed = self.model.predict(text_list)
                            doc_embed = doc_embed[:, 0, :].squeeze(0)
                            split_doc_cache[doc2id[str(doc)], :] = doc_embed
                        
                        else:
                            not_found+=1
                        
                        if count % 500 == 0:
                            print("{} done..".format(count))
                            
                else:
                    if split == 'train':
                        loader = self.config['train_loader']
                    elif split == 'val':
                        loader = self.config['dev_loader']
                    else:
                        loader = self.config['test_loader']
                    
                    for count, batch in enumerate(loader):
                        embeds = self.model(batch.text[0].to(device), batch.text[1].to(device), cache=True)
                        split_doc_cache[doc2id['gossipcop-'+str(batch.id.item())], :] = embeds[:]
                        # ids = list(batch.id)
                        # for i in range(len(ids)):
                        #     split_doc_cache[doc2id['gossipcop-'+str(ids[i].item())], :] = embeds[i, :]
                        
                        # row_sum = split_doc_cache.sum(1)
                        # row_sum = list(row_sum)
                        # c=0
                        # for s in row_sum:
                        #     if s==0:
                        #         c+=1
                        # print("Zero entries = ", c)
                        if count % 500 == 0:
                            print("{} done..".format(count))


                row_sum = split_doc_cache.sum(1)
                row_sum = list(row_sum)
                c=0
                for s in row_sum:
                    if s==0:
                        c+=1
                print("Zero entries = ", c)
                print("count = ", count)
                print("Not found = ", not_found)
                name = 'doc_embeds_roberta' if self.config['embed_name'] == 'roberta' else 'doc_embeds_cnn'
                doc_embed_file = os.path.join(self.comp_dir, 'cached_embeds', '{}_{}_{}.pt'.format(name, self.config['seed'], split))
                print("\nSaving docs embeddings in : ", doc_embed_file)
                torch.save(split_doc_cache, doc_embed_file)
                # loaded_embeds = torch.load(doc_embed_file)
                
                
                
                
                
    def predict_and_cache_fakehealth(self):
        print("\n\nCaching FakeHealth dataset : ", self.config['data_name'])
        docs_splits_file = os.path.join(self.data_dir, 'doc_splits_{}.json'.format(self.config['data_name']))
        doc2labels_file = os.path.join(self.data_dir, 'doc2labels.json')
        doc_splits = json.load(open(docs_splits_file, 'r'))
        doc2labels = json.load(open(doc2labels_file, 'r'))
        test_docs = doc_splits['test_docs']
        train_docs = doc_splits['train_docs']
        val_docs = doc_splits['val_docs']
    
        # Creating doc2id dictionary
        print("\nCreating doc2id dictionary..")
        doc2id = {}
        src_dir = os.path.join(self.data_dir, 'content', self.config['data_name'])
        for root, dirs, files in os.walk(src_dir):
            for count,file in enumerate(files):
                doc= file.split('.')[0]
                # if str(doc) in test_docs:
                doc2id[str(doc)] = len(doc2id)
        
        name = 'doc2id_encoder.json' if self.config['embed_name'] == 'roberta' else 'doc2id_cnn_encoder.json'
        doc2id_file = os.path.join(self.comp_dir, name)
        print("doc2id = ", len(doc2id))
        print("Saving in : ", doc2id_file)
        with open(doc2id_file, 'w+') as json_file:
            json.dump(doc2id, json_file)                 
            
        if self.config['embed_name'] == 'glove':
            self.config['batch_size'] = 1
            self.config['train_loader'], self.config['dev_loader'], self.config['test_loader'], self.config['TEXT'], self.config['LABEL'], self.config['train_split'], self.config['val_split'], self.config['test_split'] = prepare_glove_training(self.config)
        elif self.config['embed_name'] == 'elmo':
            self.config['batch_size'] = 1
            self.config['train_data'], self.config['train_labels'], self.config['val_data'], self.config['val_labels'], self.config['test_data'], self.config['test_labels'] = prepare_elmo_training(self.config, fold=0, verbose=False)                 
            
        
        # iterating over test_docs and saving their representations
        splits = ['train', 'val', 'test']
        # print("\nIterating over doc2id and saving their representations...")
        for split in splits:
            print("\nObtaining {} doc representations...".format(split))
            if split == 'train':
                split_docs = train_docs
            elif split == 'val':
                split_docs = val_docs
            else:
                split_docs = test_docs
            
            embed_dim = 1024 if self.config['embed_name'] == 'roberta' else 384
            split_doc_cache = torch.zeros(len(doc2id), embed_dim).to(device)
            print("split_doc_cache shape = ", split_doc_cache.shape)
            not_found=0
            with torch.no_grad():
                if self.config['embed_name'] == 'roberta':
                    for count, doc in enumerate(split_docs):
                        doc_file = os.path.join(self.data_dir, 'content', self.config['data_name'], str(doc)+'.json')
                        if os.path.isfile(doc_file):
                            text_list= []
                            text = json.load(open(doc_file, 'r') )
                            text = text['text'].replace('\n', ' ')
                            text = text.replace('\t', ' ')
                            text_list.append(str(text))
                            _, _, doc_embed = self.model.predict(text_list)
                            doc_embed = doc_embed[:, 0, :].squeeze(0)
                            split_doc_cache[doc2id[str(doc)], :] = doc_embed
                            
                            
                            if count % 500 == 0:
                                print("{} done..".format(count))
                        else:
                            not_found+=1
                
                else:
                    if split == 'train':
                        loader = self.config['train_loader']
                    elif split == 'val':
                        loader = self.config['dev_loader']
                    else:
                        loader = self.config['test_loader']
                    
                    for count, batch in enumerate(loader):
                        embeds = self.model(batch.text[0].to(device), batch.text[1].to(device), cache=True)
                        split_doc_cache[doc2id['story_reviews_'+str(batch.id.item())], :] = embeds[:]
                        # ids = list(batch.id)
                        # for i in range(len(ids)):
                        #     split_doc_cache[doc2id['gossipcop-'+str(ids[i].item())], :] = embeds[i, :]
                        
                        if count % 500 == 0:
                            print("{} done..".format(count))
                
                row_sum = split_doc_cache.sum(1)
                row_sum = list(row_sum)
                c=0
                for s in row_sum:
                    if s==0:
                        c+=1
                print("Zero entries = ", c)
                print("count = ", count)
                print("Not found = ", not_found)
                name = 'doc_embeds_roberta' if self.config['embed_name'] == 'roberta' else 'doc_embeds_cnn'
                doc_embed_file = os.path.join(self.comp_dir, 'cached_embeds', '{}_{}_{}.pt'.format(name, self.config['seed'], split))
                print("\nSaving docs embeddings in : ", doc_embed_file)
                torch.save(split_doc_cache, doc_embed_file)
                # loaded_embeds = torch.load(doc_embed_file)