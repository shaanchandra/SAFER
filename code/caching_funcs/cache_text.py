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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Cache_Text_Embeds():
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.comp_dir = os.path.join('data', 'complete_data', config['data_name'])
        self.data_dir = os.path.join('FakeHealth')
        
        if self.config['embed_name'] == 'glove':
            self.config['batch_size'] = 1
            self.config['train_loader'], self.config['dev_loader'], self.config['test_loader'], self.config['TEXT'], self.config['LABEL'], self.config['train_split'], self.config['val_split'], self.config['test_split'] = prepare_glove_training(self.config)
        elif self.config['embed_name'] == 'elmo':
            self.config['batch_size'] = 1
            self.config['train_data'], self.config['train_labels'], self.config['val_data'], self.config['val_labels'], self.config['test_data'], self.config['test_labels'] = prepare_elmo_training(self.config, fold=0, verbose=False)                 

        
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
    
    
    
    
    def obtain_cnn_representations(self):
        
        prefix = 'gossipcop-' if self.config['data_name'] == 'gossipcop' else 'story_reviews_'
        with torch.no_grad():
            if self.split == 'train':
                loader = self.config['train_loader']
            elif self.split == 'val':
                loader = self.config['dev_loader']
            else:
                loader = self.config['test_loader']
            
            for count, batch in enumerate(loader):
                embeds = self.model(batch.text[0].to(device), batch.text[1].to(device), cache=True)
                self.split_doc_cache[self.doc2id[prefix + str(batch.id.item())], :] = embeds[:]
                # ids = list(batch.id)
                # for i in range(len(ids)):
                #     split_doc_cache[doc2id['gossipcop-'+str(ids[i].item())], :] = embeds[i, :]
                
                if self.count % 500 == 0:
                    print("{} done..".format(count))
    
    


    
    def obtain_roberta_representations(self):
        
        with torch.no_grad():
            for self.count, self.doc in enumerate(self.split_docs):
                if self.config['data_name'] in ['gossipcop', 'politifact']:
                    real_path = os.path.join('data', 'base_data', self.config['data_name'], 'real', str(self.doc)+'.json')
                    fake_path = os.path.join('data', 'base_data', self.config['data_name'], 'fake', str(self.doc)+'.json')
                    self.doc_file = real_path if os.path.isfile(real_path) else fake_path
                else:
                    self.doc_file = os.path.join(self.data_dir, 'content', self.config['data_name'], str(self.doc)+'.json')
        
                if os.path.isfile(self.doc_file):
                    text_list= []
                    text = json.load(open(self.doc_file, 'r') )
                    text = text['text'].replace('\n', ' ')
                    text = text.replace('\t', ' ')
                    text_list.append(str(text))
                    _, _, doc_embed = self.model.predict(text_list)
                    doc_embed = doc_embed[:, 0, :].squeeze(0)
                    self.split_doc_cache[self.doc2id[str(self.doc)], :] = doc_embed
                
                else:
                    self.not_found+=1
                
                if self.count % 500 == 0:
                    print("{} done..".format(self.count))
    
    
    
    
    
    
    def obtain_doc_representations(self):
        
        # iterating over test_docs and saving their representations
        splits = ['train', 'val', 'test']
        for self.split in splits:
            print("\nObtaining {} doc representations...".format(self.split))
            if self.split == 'train':
                self.split_docs = self.train_docs
            elif self.split == 'val':
                self.split_docs = self.val_docs
            else:
                self.split_docs = self.test_docs
            
            embed_dim = 1024 if self.config['embed_name'] == 'roberta' else 384
            self.split_doc_cache = torch.zeros(len(self.doc2id), embed_dim).to(device)
            print("split_doc_cache shape = ", self.split_doc_cache.shape)
            self.not_found=0
            
            if self.config['embed_name'] == 'roberta':
                self.obtain_roberta_representations()
            else:
                self.obtain_cnn_representations()
                    

            row_sum = self.split_doc_cache.sum(1)
            row_sum = list(row_sum)
            c=0
            for s in row_sum:
                if s==0:
                    c+=1
            print("Zero entries = ", c)
            print("count = ", self.count)
            print("Not found = ", self.not_found)
            name = 'doc_embeds_{}'.format(self.config['embed_name'])
            doc_embed_file = os.path.join(self.comp_dir, 'cached_embeds', '{}_{}_{}.pt'.format(name, self.config['seed'], self.split))
            print("\nSaving docs embeddings in : ", doc_embed_file)
            torch.save(self.split_doc_cache, doc_embed_file)
            # loaded_embeds = torch.load(doc_embed_file)
    
    
    
    
               
            
