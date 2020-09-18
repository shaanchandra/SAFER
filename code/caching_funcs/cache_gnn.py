import torch
import sys, os, json
import warnings
warnings.filterwarnings("ignore")

import nltk
nltk.download('punkt')
sys.path.append("..")
from torch_geometric.utils import to_dense_adj

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   



class Cache_GNN_Embeds():
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.model_file = config['model_file']
        self.comp_dir = os.path.join('data', 'complete_data', config['data_name'])
        self.data_dir = os.path.join('FakeHealth')

        
        if config['data_name'] in ['gossipcop', 'politifcat']:
            self.predict_and_cache_fakenews()
        elif config['data_name'] in ['HealthStory', 'HealthRelease']:
            self.predict_and_cache_fakehealth()
            

    
    def predict_and_cache_fakenews(self):        
        print("\n\nCaching FakeNews dataset : ", self.config['data_name'])
        checkpoint = torch.load(self.model_file)
        self.model.load_state_dict(checkpoint['model_state_dict'])  
        

        node2id_file = os.path.join(self.comp_dir, 'node2id_lr_30_30.json')
        docs_splits_file = os.path.join(self.comp_dir, 'doc_splits_lr.json')
        self.node2id = json.load(open(node2id_file, 'r'))
        self.doc_splits = json.load(open(docs_splits_file, 'r'))
        self.train_docs = self.doc_splits['train_docs']
        self.val_docs = self.doc_splits['val_docs']
        self.test_docs = self.doc_splits['test_docs']        
        
        self.prepare_id2node()
        self.obtain_user_representations()    
        self.obtain_doc_representations()
        
        
        
    
    def predict_and_cache_fakehealth(self):
        
        print("\n\nCaching FakeHealth dataset : ", self.config['data_name'])
        checkpoint = torch.load(self.model_file)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        node2id_file = os.path.join(self.comp_dir, 'node2id_lr_top10.json')
        self.node2id = json.load(open(node2id_file, 'r'))
        docs_splits_file = os.path.join(self.data_dir, 'doc_splits_{}.json'.format(self.config['data_name']))
        self.doc_splits = json.load(open(docs_splits_file, 'r'))
        
        self.train_docs = self.doc_splits['train_docs']
        self.val_docs = self.doc_splits['val_docs']
        self.test_docs = self.doc_splits['test_docs']
                
        self.prepare_id2node()
        self.obtain_user_representations()    
        self.obtain_doc_representations()
        

    
    def prepare_id2node(self):
        print("\n\nPreparing id2node dict..")
        self.id2node = {}
        for node, idx in self.node2id.items():
            self.id2node[int(idx)] = node
    

    
    def obtain_user_representations(self):
        print("\nObtaining user representations...")
        self.user_cache = torch.zeros(len(self.node2id), self.config['embed_dim']) if self.config['model_name'] not in ['gat', 'rgat'] else torch.zeros(len(self.node2id), 3*self.config['embed_dim'])
        
        with torch.no_grad(): 
            self.model.eval()
            if self.config['full_graph']:
                _, _, node_embeds = self.model(self.config['data'].x.to(device), self.config['data'].edge_index.to(device))
                for idx, ids in enumerate(self.node2id.values()):
                    node = self.id2node[int(ids)]
                    if not str(node).startswith('g'):
                        self.user_cache[ids, :] = node_embeds[ids, :]
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
                        node = self.id2node[int(ids)]
                        if not str(node).startswith('g'):
                            self.user_cache[ids, :] = node_embeds[idx, :]
    
        
        name = 'user_embeds_graph_lr' if self.config['data_name'] in ['HealthRelease', 'HealthStory'] else 'user_embeds_graph_lr_30_30'
        user_embed_file = os.path.join(self.comp_dir, 'cached_embeds', '{}_{}_{}.pt'.format(name, self.config['seed'], self.config['model_name']))
        print("\nSaving user embeddings in : ", user_embed_file)
        torch.save(self.user_cache, user_embed_file)
        
                
        
        
    def obtain_doc_representations(self):
        splits = ['train', 'val', 'test']
        for split in splits:
            print("\nObtaining {} doc representations...".format(split))
            if split == 'train':
                split_docs = self.train_docs
            elif split == 'val':
                split_docs = self.val_docs
            else:
                split_docs = self.test_docs
            
            split_doc_cache = torch.zeros(len(self.node2id), self.config['embed_dim']) if self.config['model_name'] != 'gat' else torch.zeros(len(self.node2id), 3*self.config['embed_dim'])
            print("split_doc_cache shape = ", split_doc_cache.shape)
            for count, doc in enumerate(split_docs):
                if self.config['data_name'] in ['gossipcop', 'politifact']:
                    doc_file = os.path.join(self.comp_dir, 'complete', str(doc)+'.json')
                    users = json.load(open(doc_file, 'r'))['users']
                else:
                    doc_file = os.path.join(self.data_dir, 'engagements', 'complete', self.config['data_name'], str(doc)+'.json')
                    if os.path.isfile(doc_file):
                        users = json.load(open(doc_file, 'r'))['users'] 
            
                c=0
                for user in users:
                    if str(user) in self.node2id:
                        c+=1
                        split_doc_cache[self.node2id[str(doc)], : ] += self.user_cache[self.node2id[str(user)]]
                
                if c>0:
                    split_doc_cache[self.node2id[str(doc)], : ] /=  c # normalize the sum by no. of users 
                if count% 500 == 0:
                    print("{} done...".format(count))

            
            name = 'doc_embeds_graph_poinc_wt3_lr' if self.config['data_name'] in ['HealthRelease', 'HealthStory'] else 'doc_embeds_graph_lr_30_30_poinc'
            doc_embed_file = os.path.join(self.comp_dir, 'cached_embeds', '{}_{}_{}_{}.pt'.format(name, split, self.config['seed'], self.config['model_name']))
            print("\nSaving doc embeddings in : ", doc_embed_file)
            torch.save(split_doc_cache, doc_embed_file)
            # loaded_embeds = torch.load(doc_embed_file)

