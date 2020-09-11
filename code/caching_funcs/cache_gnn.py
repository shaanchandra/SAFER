import torch
import torch.nn.functional as F
import argparse, time, datetime, shutil
import sys, os, glob, json, random
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import nltk
nltk.download('punkt')
sys.path.append("..")
from torch_geometric.utils import to_dense_adj
# from utils.utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   



class Cache_GNN_Embeds():
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.model_file = config['model_file']
        self.comp_dir = os.path.join('data', 'complete_data', config['data_name'])
        self.data_dir = os.path.join('FakeHealth')

        
        if config['data_name'] == 'pheme':
            self.predict_and_cache_pheme()
        elif config['data_name'] in ['gossipcop', 'politifcat']:
            self.predict_and_cache_fakenews()
        elif config['data_name'] in ['HealthStory', 'HealthRelease']:
            self.predict_and_cache_fakehealth()
            
            
            

    def predict_and_cache_pheme(self):
        checkpoint = torch.load(self.model_file)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        base_dir = os.path.join(os.getcwd(), '..', 'data', 'complete_data', 'pheme_cv')
        node2id_file = os.path.join(base_dir, 'fold_{}'.format(self.config['fold']), 'node2id_lr_just.json')
        doc2id_file = os.path.join(base_dir, 'fold_{}'.format(self.config['fold']), 'doc2id_train_just.json')
        user2id_file = os.path.join(base_dir, 'fold_{}'.format(self.config['fold']), 'user2id_train_just.json')
        node2id = json.load(open(node2id_file, 'r'))
        docs_splits_file = os.path.join(base_dir, 'fold_{}'.format(self.config['fold']), 'doc_splits.json')
        doc_splits = json.load(open(docs_splits_file, 'r'))
        train_docs = doc_splits['train_docs']
        val_docs = doc_splits['val_docs']
        test_docs = doc_splits['test_docs']
        all_docs = train_docs + test_docs
        # N, _ = data.x.shape
        user_cache = torch.zeros(len(node2id), self.config['embed_dim']) if self.config['model_name'] != 'gat' else torch.zeros(len(node2id), 3*self.config['embed_dim'])
        print("user_cache shape = ", user_cache.shape)
        
        print("\n\nPreparing id2node dict..")
        id2node = {}
        for node, idx in node2id.items():
            id2node[int(idx)] = node
        

        print("\nObtaining user representations...")            
        with torch.no_grad(): 
            if self.config['full_graph']:
                if self.config['model_name'] in ['rgcn', 'rgat', 'rsage']:
                    _, _, node_embeds = self.model(self.config['data'].x.to(device), self.config['data'].edge_index.to(device), self.config['data'].edge_attr.to(device))
                elif self.config['model_name'] == 'HGCN':
                    preds = self.model.encode(self.config['data'].x.to(device), self.config['data'].edge_index.to(device))
                    _, node_embeds = self.model.decode(preds.to(device), self.config['data'].edge_index.to(device))
                else:
                    _, _, node_embeds = self.model(self.config['data'].x.to(device), self.config['data'].edge_index.to(device))
                for idx, ids in enumerate(node2id.values()):
                    node = id2node[int(ids)]
                    if str(node) not in all_docs:
                        user_cache[ids, :] = node_embeds[ids, :]
            else:
                for iters, batch_data in enumerate(self.config['loader']): 
                    if self.config['model_name'] in ['rgcn', 'rgat', 'rsage']:
                        _, _, node_embeds = self.model(batch_data.x.to(device), batch_data.edge_index.to(device), batch_data.edge_attr.to(device))
                    elif self.config['model_name'] == 'HGCN':
                        batch_data.edge_index = to_dense_adj(batch_data.edge_index).squeeze(0)
                        preds = self.model.encode(batch_data.x.to(device), batch_data.edge_index.to(device))
                        _, node_embeds = self.model.decode(preds.to(device), batch_data.edge_index.to(device))
                        
                    else:
                        _, _, node_embeds = self.model(batch_data.x.to(device), batch_data.edge_index.to(device))
                    for idx, ids in enumerate(batch_data.node2id):
                        node = id2node[int(ids)]
                        if str(node) not in all_docs:
                            user_cache[ids, :] = node_embeds[idx, :]

        row_sum = user_cache.sum(1)
        row_sum = list(row_sum)
        c=0
        for s in row_sum:
            if s==0:
                c+=1
        print("Zero entries = ", c)
        
        user_embed_file = os.path.join(base_dir, 'fold_{}'.format(self.config['fold']), 'user_embeds_graph_lr_{}_{}.pt'.format(self.config['seed'], self.config['model_name']))
        print("\nSaving user embeddings in : ", user_embed_file)
        #torch.save(user_cache, user_embed_file)
        
        splits = ['train', 'test']
        for split in splits:
            print("\nObtaining {} doc representations...".format(split))
            if split == 'train':
                split_docs = train_docs
            else:
                split_docs = test_docs
            
            split_doc_cache = torch.zeros(len(node2id), self.config['embed_dim']) if self.config['model_name'] != 'gat' else torch.zeros(len(node2id), 3*self.config['embed_dim'])
            print("split_doc_cache shape = ", split_doc_cache.shape)
            for count, doc in enumerate(split_docs):
                doc_file = os.path.join(base_dir, 'complete', str(doc)+'.json')
                user = json.load(open(doc_file, 'r'))['source_user']
                if str(user) in node2id and str(doc) in node2id:
                    split_doc_cache[node2id[str(doc)], : ] = user_cache[node2id[str(user)]]
                
                if count% 500 == 0:
                    print("{} done...".format(count))
            
            row_sum = split_doc_cache.sum(1)
            row_sum = list(row_sum)
            c=0
            for s in row_sum:
                if s==0:
                    c+=1
            print("Zero entries = ", c)

            doc_embed_file = os.path.join(base_dir, 'fold_{}'.format(self.config['fold']), 'doc_embeds_graph_lr_poinc_{}_{}_{}.pt'.format(split, self.config['seed'], self.config['model_name']))
            print("\nSaving doc embeddings in : ", doc_embed_file)
            torch.save(split_doc_cache, doc_embed_file)
            # loaded_embeds = torch.load(doc_embed_file)
        
    
 
    
    def predict_and_cache_fakenews(self):
        print("\n\nCaching FakeNews dataset : ", self.config['data_name'])
        checkpoint = torch.load(self.model_file)
        self.model.load_state_dict(checkpoint['model_state_dict'])  
        
        #config['device'] = 'cpu'
        #device = 'cpu'
        node2id_file = os.path.join(self.comp_dir, 'node2id_lr_30_30.json')
        doc2id_file = os.path.join(self.comp_dir, 'doc2id_lr_train_30_30.json')
        user2id_file = os.path.join(self.comp_dir, 'user2id_lr_train_30_30.json')
        node2id = json.load(open(node2id_file, 'r'))
        docs_splits_file = os.path.join(self.comp_dir, 'doc_splits_lr.json')
        doc_splits = json.load(open(docs_splits_file, 'r'))
        train_docs = doc_splits['train_docs']
        val_docs = doc_splits['val_docs']
        test_docs = doc_splits['test_docs']
        # N, _ = data.x.shape
        user_cache = torch.zeros(len(node2id), self.config['embed_dim']) if self.config['model_name'] != 'gat' else torch.zeros(len(node2id), 3*self.config['embed_dim'])
        
        print("\n\nPreparing id2node dict..")
        id2node = {}
        for node, idx in node2id.items():
            id2node[int(idx)] = node
        
    
        print("\nObtaining user representations...")            
        with torch.no_grad(): 
            self.model.eval()
            if self.config['full_graph']:
                _, _, node_embeds = self.model(self.config['data'].x.to(device), self.config['data'].edge_index.to(device))
                for idx, ids in enumerate(node2id.values()):
                    node = id2node[int(ids)]
                    if not str(node).startswith('g'):
                        user_cache[ids, :] = node_embeds[ids, :]
            else:
                for iters, batch_data in enumerate(self.config['loader']): 
                    if self.config['model_name'] in ['rgcn', 'rgat', 'rsage']:
                        _, _, node_embeds = self.model(batch_data.x.to(device), batch_data.edge_index.to(device), batch_data.edge_attr.to(device))
                    elif self.config['model_name'] == 'HGCN':
                        batch_data.edge_index = to_dense_adj(batch_data.edge_index).squeeze(0)
                        preds = self.model.encode(batch_data.x.to(device), batch_data.edge_index.to(device))
                        _, node_embeds = self.model.decode(preds.to(device), batch_data.edge_index.to(device))
                    else:
                        _, _, node_embeds = self.model(batch_data.x.to(device), batch_data.edge_index.to(device))
                        
                    for idx, ids in enumerate(batch_data.node2id):
                        node = id2node[int(ids)]
                        if not str(node).startswith('g'):
                            user_cache[ids, :] = node_embeds[idx, :]
    
        row_sum = user_cache.sum(1)
        row_sum = list(row_sum)
        c=0
        for s in row_sum:
            if s==0:
                c+=1
        print("Zero entries = ", c)
        
        user_embed_file = os.path.join(self.comp_dir, 'cached_embeds', 'user_embeds_graph_lr_30_30_{}_{}.pt'.format(self.config['seed'], self.config['model_name']))
        print("\nSaving user embeddings in : ", user_embed_file)
        torch.save(user_cache, user_embed_file)
        
        
        
        splits = ['train', 'val', 'test']
        for split in splits:
            print("\nObtaining {} doc representations...".format(split))
            if split == 'train':
                split_docs = train_docs
            elif split == 'val':
                split_docs = val_docs
            else:
                split_docs = test_docs
            
            split_doc_cache = torch.zeros(len(node2id), self.config['embed_dim']) if self.config['model_name'] != 'gat' else torch.zeros(len(node2id), 3*self.config['embed_dim'])
            print("split_doc_cache shape = ", split_doc_cache.shape)
            for count, doc in enumerate(split_docs):
                doc_file = os.path.join(self.comp_dir, 'complete', str(doc)+'.json')
                users = json.load(open(doc_file, 'r'))['users']
                c=0
                for user in users:
                    if str(user) in node2id:
                        c+=1
                        split_doc_cache[node2id[str(doc)], : ] += user_cache[node2id[str(user)]]
                
                if c>0:
                    split_doc_cache[node2id[str(doc)], : ] /=  c # normalize the sum by no. of users 
                if count% 2000 == 0:
                    print("{} done...".format(count))
            
            row_sum = split_doc_cache.sum(1)
            row_sum = list(row_sum)
            c=0
            for s in row_sum:
                if s==0:
                    c+=1
            print("Zero entries = ", c)
    
            doc_embed_file = os.path.join(self.comp_dir, 'cached_embeds', 'doc_embeds_graph_lr_30_30_poinc_{}_{}_{}.pt'.format(split, self.config['seed'], self.config['model_name']))
            print("\nSaving doc embeddings in : ", doc_embed_file)
            torch.save(split_doc_cache, doc_embed_file)
            # loaded_embeds = torch.load(doc_embed_file)
    
    
    
    
    def predict_and_cache_fakehealth(self):
        # Seeds for reproduceable runs
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        random.seed(self.config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        print("\n\nCaching FakeHealth dataset : ", self.config['data_name'])
        checkpoint = torch.load(self.model_file)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        node2id_file = os.path.join(self.comp_dir, 'node2id_lr_top10.json')
        doc2id_file = os.path.join(self.comp_dir, 'doc2id_lr_top10_train.json')
        user2id_file = os.path.join(self.comp_dir, 'user2id_lr_top10_train.json')
        user_type_file = os.path.join(self.comp_dir, 'user_types.json')
        node2id = json.load(open(node2id_file, 'r'))
        docs_splits_file = os.path.join(self.data_dir, 'doc_splits_{}.json'.format(self.config['data_name']))
        doc_splits = json.load(open(docs_splits_file, 'r'))
        user_types = json.load(open(user_type_file, 'r'))
        
        train_docs = doc_splits['train_docs']
        val_docs = doc_splits['val_docs']
        test_docs = doc_splits['test_docs']
        # N, _ = data.x.shape
        user_cache = torch.zeros(len(node2id), self.config['embed_dim']) if self.config['model_name'] != 'gat' else torch.zeros(len(node2id), 3*self.config['embed_dim'])
        print(user_cache.shape)
        
        print("\n\nPreparing id2node dict..")
        id2node = {}
        for node, idx in node2id.items():
            id2node[int(idx)] = node
        
    
        print("\nObtaining user representations...")            
        with torch.no_grad(): 
            self.model.eval()
            if self.config['full_graph']:
                if self.config['model_name'] not in ['HGCN', 'HNN']:
                    _, _, node_embeds = self.model(self.config['data'].x.to(device), self.config['data'].edge_index.to(device))
                else:
                    preds = self.model.encode(self.config['data'].x.to(device), self.config['data'].edge_index.to(device))
                    _, node_embeds = self.model.decode(preds, self.config['data'].edge_index.to(device))
                for idx, ids in enumerate(node2id.values()):
                    node = id2node[int(ids)]
                    if not str(node).startswith('news') and not str(node).startswith('story'): # and (str(node) in user_types['only_fake'] or str(node) in user_types['only_real']):
                        user_cache[ids, :] = node_embeds[ids, :]
            else:
                for iters, batch_data in enumerate(self.config['loader']): 
                    if self.config['model_name'] in ['rgcn', 'rgat', 'rsage']:
                        _, _, node_embeds = self.model(batch_data.x.to(device), batch_data.edge_index.to(device), batch_data.edge_attr.to(device))
                        
                    elif self.config['model_name'] == 'HGCN':
                        batch_data.edge_index = to_dense_adj(batch_data.edge_index).squeeze(0)
                        preds = self.model.encode(batch_data.x.to(device), batch_data.edge_index.to(device))
                        _, node_embeds = self.model.decode(preds.to(device), batch_data.edge_index.to(device))
                    else:
                        _, _, node_embeds = self.model(batch_data.x.to(device), batch_data.edge_index.to(device))
                        
                    for idx, ids in enumerate(batch_data.node2id):
                        node = id2node[int(ids)]
                        if not str(node).startswith('news') and not str(node).startswith('story'):
                            user_cache[ids, :] = node_embeds[idx, :]
    
        row_sum = user_cache.sum(1)
        row_sum = list(row_sum)
        c=0
        for s in row_sum:
            if s==0:
                c+=1
        print("Zero entries = ", c)
        
        user_embed_file = os.path.join(self.comp_dir, 'cached_embeds', 'user_embeds_graph_lr_{}_{}.pt'.format(self.config['seed'], self.config['model_name']))
        print("\nSaving user embeddings in : ", user_embed_file)
        # torch.save(user_cache, user_embed_file)
        
        
        
        splits = ['train', 'val', 'test']
        for split in splits:
            print("\nObtaining {} doc representations...".format(split))
            if split == 'train':
                split_docs = train_docs
            elif split == 'val':
                split_docs = val_docs
            else:
                split_docs = test_docs
            
            split_doc_cache = torch.zeros(len(node2id), self.config['embed_dim']) if self.config['model_name'] != 'gat' else torch.zeros(len(node2id), 3*self.config['embed_dim'])
            print("split_doc_cache shape = ", split_doc_cache.shape)
            for count, doc in enumerate(split_docs):
                doc_file = os.path.join(self.data_dir, 'engagements', 'complete', self.config['data_name'], str(doc)+'.json')
                if os.path.isfile(doc_file):
                    users = json.load(open(doc_file, 'r'))['users']
                    c=0
                    for user in users:
                        if str(user) in node2id:
                            c+=1
                            split_doc_cache[node2id[str(doc)], : ] += user_cache[node2id[str(user)]]
                    
                    if c>0:
                        split_doc_cache[node2id[str(doc)], : ] /=  c # normalize the sum by no. of users 
                    if count% 2000 == 0:
                        print("{} done...".format(count))
            
            row_sum = split_doc_cache.sum(1)
            row_sum = list(row_sum)
            c=0
            for s in row_sum:
                if s==0:
                    c+=1
            print("Zero entries = ", c)
    
            doc_embed_file = os.path.join(self.comp_dir, 'cached_embeds', 'doc_embeds_graph_poinc_wt3_lr_{}_{}_{}.pt'.format(split, self.config['seed'], self.config['model_name']))
            print("\nSaving doc embeddings in : ", doc_embed_file)
            torch.save(split_doc_cache, doc_embed_file)
            # loaded_embeds = torch.load(doc_embed_file)


                
                
                
    
    