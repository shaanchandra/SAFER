import sys, os, json
import warnings
warnings.filterwarnings("ignore")

import torch
import nltk
nltk.download('punkt')
sys.path.append("..")

from models.model import Document_Classifier
from models.transformer_model import *
from utils.utils import *
from utils.data_utils_gnn import *
from utils.data_utils_txt import *
from utils.data_utils_hygnn import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Cache_Text_Embeds():
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.comp_dir = os.path.join('data', 'complete_data', config['data_name'])
        self.data_dir = os.path.join('FakeHealth')
        self.config['batch_size'] = 1
        
        prep_data = Prepare_Dataset(config)
        if self.config['embed_name'] == 'glove':
            self.config['train_loader'], self.config['val_loader'], self.config['test_loader'], self.config['TEXT'] = prep_data.prepare_glove_training()
        elif self.config['embed_name'] == 'elmo':
            self.config['train_data'], self.config['train_labels'], self.config['val_data'], self.config['val_labels'], self.config['test_data'], self.config['test_labels'] = prep_data.prepare_elmo_training()    
        elif config['embed_name'] in ['dbert', 'xlnet', 'roberta']:
            config['train_loader'], config['val_loader'], config['test_loader'] = prep_data.prepare_transformer_training()

        
        if config['data_name'] in ['gossipcop', 'politifcat']:
            self.predict_and_cache_fakenews()
        elif config['data_name'] in ['HealthStory', 'HealthRelease']:
            self.predict_and_cache_fakehealth()
                        
    
    
    
    def predict_and_cache_fakenews(self):
        print("\n\nCaching FakeNews dataset : ", self.config['data_name'])
        docs_splits_file = os.path.join(self.comp_dir, 'doc_splits_lr.json')
        self.doc_splits = json.load(open(docs_splits_file, 'r'))
        self.test_docs = self.doc_splits['test_docs']
        self.train_docs = self.doc_splits['train_docs']
        self.val_docs = self.doc_splits['val_docs']
        
        self.prepare_doc2id()
        self.obtain_doc_representations()
        

            
    def predict_and_cache_fakehealth(self):
        print("\n\nCaching FakeHealth dataset : ", self.config['data_name'])
        docs_splits_file = os.path.join(self.data_dir, 'doc_splits_{}.json'.format(self.config['data_name']))
        self.doc_splits = json.load(open(docs_splits_file, 'r'))
        self.test_docs = self.doc_splits['test_docs']
        self.train_docs = self.doc_splits['train_docs']
        self.val_docs = self.doc_splits['val_docs']
    
        self.prepare_doc2id()
        self.obtain_doc_representations()    

                
                        
    def prepare_doc2id(self):        
        # Creating doc2id dictionary
        print("\nCreating doc2id dictionary..")
        self.doc2id = {}
        if self.config['data_name'] in ['gossipcop', 'politifact']:
            splits = ['fake', 'real']
            for split in splits:
                src_dir = os.path.join('data', 'base_data', self.config['data_name'], split)
                for root, dirs, files in os.walk(src_dir):
                    for count,file in enumerate(files):
                        doc= file.split('.')[0]
                        # if str(doc) in test_docs:
                        self.doc2id[str(doc)] = len(self.doc2id)
        
        else:
            src_dir = os.path.join(self.data_dir, 'content', self.config['data_name'])
            for root, dirs, files in os.walk(src_dir):
                for count,file in enumerate(files):
                    doc= file.split('.')[0]
                    # if str(doc) in test_docs:
                    self.doc2id[str(doc)] = len(self.doc2id)  
        
        name = 'doc2id_{}_encoder.json'.format(self.config['embed_name']) 
        doc2id_file = os.path.join(self.comp_dir, name)
        print("doc2id = ", len(self.doc2id))
        print("Saving in : ", doc2id_file)
        with open(doc2id_file, 'w+') as json_file:
            json.dump(self.doc2id, json_file)             
    
    
  
    
    def obtain_doc_representations(self):        
        # iterating over test_docs and saving their representations
        splits = ['train', 'val', 'test']
        with torch.no_grad():
            for self.split in splits:
                print("\nObtaining {} doc representations...".format(self.split))
                if self.split == 'train':
                    loader = self.config['train_loader']
                elif self.split == 'val':
                    loader = self.config['val_loader']
                else:
                    loader = self.config['test_loader']
                
                embed_dim = 1024 if self.config['embed_name'] == 'roberta' else 384
                self.split_doc_cache = torch.zeros(len(self.doc2id), embed_dim).to(device)
                print("split_doc_cache shape = ", self.split_doc_cache.shape)
                self.not_found=0
                
                for count, batch in enumerate(loader):
                    if self.config['embed_name'] == 'cnn':
                        embeds = self.model(batch.text[0].to(device), batch.text[1].to(device), cache=True)
                        self.split_doc_cache[self.doc2id[prefix + str(batch.id.item())], :] = embeds[:]
                        # ids = list(batch.id)
                        # for i in range(len(ids)):
                        #     split_doc_cache[doc2id['gossipcop-'+str(ids[i].item())], :] = embeds[i, :]
                    
                    elif config['embed_name'] in ['dbert', 'xlnet', 'roberta']:
                        embeds = self.model(inp=batch['input_ids'], attn_mask=batch['attention_mask'])
                        self.split_doc_cache[self.doc2id[prefix + str(batch['ids'].item())], :] = embeds[:]
                        
                    if self.count % 500 == 0:
                        print("{} done..".format(count))
                    
                
                print("count = ", self.count)
                print("Not found = ", self.not_found)
                name = 'doc_embeds_{}'.format(self.config['embed_name'])
                doc_embed_file = os.path.join(self.comp_dir, 'cached_embeds', '{}_{}_{}.pt'.format(name, self.config['seed'], self.split))
                print("\nSaving docs embeddings in : ", doc_embed_file)
                torch.save(self.split_doc_cache, doc_embed_file)
                # loaded_embeds = torch.load(doc_embed_file)
    
    
    
    
               
            
