import json, time, random, datetime
import os, sys, glob, csv, re, argparse
import numpy as np
import networkx as nx
from collections import defaultdict
from scipy.sparse import csr_matrix, lil_matrix, save_npz, load_npz
from collections import defaultdict

import nltk
# nltk.download('punkt')
from nltk.corpus import stopwords
sys.path.append("..")


class GNN_PreProcess():
    def __init__(self, config):
        self.config = config
        self.data_dir = config['data_dir']
        self.comp_dir = os.path.join('..', 'data', 'complete_data')
        self.datasets = []
        
    
    def get_label_distribution(self, labels):  
        fake = labels.count(1)
        real = labels.count(0)
        denom = fake+real
        return fake/denom, real/denom
    


    def calc_elapsed_time(self, start, end):
        hours, rem = divmod(end-start, 3600)
        time_hours, time_rem = divmod(end, 3600)
        minutes, seconds = divmod(rem, 60)
        time_mins, _ = divmod(time_rem, 60)
        return int(hours), int(minutes), int(seconds)
    
    
        
    
    def save_adj_matrix(self, dataset, adj_matrix, edge_type, adj_file, edge_type_file):
        
        adj_file = os.path.join(self.comp_dir, dataset, adj_file)
        print("\nMatrix construction done! Saving in  {}".format(adj_file+'.npz'))
        save_npz(adj_file, adj_matrix.tocsr())
        # np.save(filename, adj_matrix)
        
        edge_type_file = os.path.join(self.comp_dir, dataset, edge_type_file)
        print("\nedge_type construction done! Saving in  {}".format(edge_type_file+'.npz'))
        save_npz(edge_type_file, edge_type.tocsr())
        
        # Creating an edge_list matrix of the adj_matrix as required by some GCN frameworks
        print("\nCreating edge_index format of adj_matrix...")
        # G = nx.DiGraph(adj_matrix.tocsr())
        # temp_matrix = adj_matrix.toarray()
        # rows, cols = np.nonzero(temp_matrix)
        rows, cols = adj_matrix.nonzero()
        
        edge_index = np.vstack((np.array(rows), np.array(cols)))
        print("Edge index shape = ", edge_index.shape)
        
        edge_matrix_file = os.path.join(self.comp_dir, dataset, adj_file+'_edge.npy')
        print("saving edge_list format in :  ", edge_matrix_file)
        np.save(edge_matrix_file, edge_index, allow_pickle=True)
        
        edge_index = edge_type[edge_type.nonzero()]
        edge_index = edge_index.toarray()
        edge_index = edge_index.squeeze(0)
        print("edge_type shape = ", edge_index.shape)
        edge_matrix_file = os.path.join(self.comp_dir, dataset, edge_type_file+'_edge.npy')
        print("saving edge_type edge list format in :  ", edge_matrix_file)
        np.save(edge_matrix_file, edge_index, allow_pickle=True)
        
        
        
        
        
    def build_vocab(self, vocab_file, dataset, train_docs, doc2id):
        vocab = {}
        stop_words = set(stopwords.words('english'))
        start = time.time()
        
        if not os.path.isfile(vocab_file):
            print("\nBuilding vocabulary...")  
            if self.dataset in ['gossipcop', 'politifact']:
                labels = ['fake', 'real']
                for label in labels:
                    src_doc_dir = os.path.join(self.data_dir, 'base_data', dataset, label)
                    for root, dirs, files in os.walk(src_doc_dir):
                        for file in files:
                            doc = file.split('.')[0]
                            if str(doc) in train_docs and str(doc) in doc2id:
                                src_file_path = os.path.join(root, file)
                                with open(src_file_path, 'r') as f:
                                    file_content = json.load(f)
                                    text = file_content['text'].lower()[:500]
                                    text = re.sub(r'#[\w-]+', 'hashtag', text)
                                    text = re.sub(r'https?://\S+', 'url', text)
                                    # text = re.sub(r"[^A-Za-z(),!?\'`]", " ", text)
                                    text = nltk.word_tokenize(text)
                                    text = [w for w in text if not w in stop_words]
                                    for token in text:
                                        if token not in vocab.keys():
                                            vocab[token] = len(vocab)
                    
        
                    hrs, mins, secs = self.calc_elapsed_time(start, time.time())
                    print("Done. Took {}hrs and {}mins and {}secs\n".format(hrs, mins, secs))
                    print("Size of vocab =  ", len(vocab))
                    print("Saving vocab for  {}  at:  {}".format(dataset, vocab_file))
                    with open(vocab_file, 'w+') as v:
                        json.dump(vocab, v)
            
            elif self.dataset in ['HealthStory', 'HelathRelease']:
                src_doc_dir = os.path.join(self.config['data_dir'], 'content', dataset+"/*.json")
                all_files = glob.glob(src_doc_dir)
                for file in all_files:
                    with open(file, 'r') as f:
                        file_content = json.load(f)
                        text = file_content['text'].replace('\n', ' ')[:1500]
                        text = text.replace('\t', ' ')
                        text = text.lower()
                        text = re.sub(r'#[\w-]+', 'hashtag', text)
                        text = re.sub(r'https?://\S+', 'url', text)
                        # text = re.sub(r"[^A-Za-z(),!?\'`]", " ", text)
                        text = nltk.word_tokenize(text)
                        text = [w for w in text if not w in stop_words]
                        for token in text:
                            if token not in vocab.keys():
                                vocab[token] = len(vocab)
                
                print("Saving vocab for  {}  at:  {}".format(dataset, vocab_file))
                with open(vocab_file, 'w+') as v:
                    json.dump(vocab, v)
            
        else:
            print("\nReading vocabulary from:  ", vocab_file)
            vocab = json.load(open(vocab_file, 'r'))
        
        return vocab
    
    
    
    
    
    
    def save_labels(self, dataset):
        self.doc2labels_file = os.path.join(self.comp_dir, dataset, self.doc2labels_file)
        print("Saving doc2labels for  {} at:  {}".format(dataset, self.doc2labels_file))
        with open(self.doc2labels_file, 'w+') as v:
            json.dump(self.doc2labels, v)
        
        labels_list = np.zeros(self.N)
        for key,value in self.doc2labels.items():
            labels_list[self.doc2id[str(key)]] = value
                      
        # Sanity Checks
        # print(sum(labels_list))
        # print(len(labels_list))
        # print(sum(labels_list[2402:]))
        # print(sum(labels_list[:2402]))
        
        self.labels_list_file = os.path.join(self.comp_dir, dataset, self.labels_list_file)
        temp_dict = {}
        temp_dict['labels_list'] = list(labels_list)
        print("Labels list construction done! Saving in :   {}".format(self.labels_list_file))
        with open(self.labels_list_file, 'w+') as v:
            json.dump(temp_dict, v)
        
        
        # Create the all_labels file
        all_labels = np.zeros(self.N)
        self.all_labels_file = os.path.join(self.comp_dir, dataset, self.all_labels_file)
        for doc in self.doc2labels.keys():
            all_labels[self.doc2id[str(doc)]] = self.doc2labels[str(doc)]
        
        temp_dict = {}
        temp_dict['all_labels'] = list(all_labels)
        print("Sum of labels this test set = ", sum(all_labels))
        print("Len of labels = ", len(all_labels))
        print("all_labels list construction done! Saving in :   {}".format(self.all_labels_file))
        with open(self.all_labels_file, 'w+') as j:
            json.dump(temp_dict, j)
    
    
    
    
    
    def create_split_masks_main(self, dataset):
        doc2id = json.load(open(self.doc2id_file, 'r'))
        doc_splits = json.load(open(self.doc_splits_file, 'r'))
        train_adj = load_npz(self.train_adj_matrix_file)
        
        train_docs = doc_splits['train_docs']
        val_docs = doc_splits['val_docs']
        
        train_n, _ = train_adj.shape
        del train_adj
        
        train_mask, val_mask = np.zeros(train_n), np.zeros(train_n) # np.zeros(test_n)
        representation_mask = np.ones(train_n)
        
        not_in_either=0
        for doc, id in doc2id.items():
            if str(doc) in train_docs:
                train_mask[id] = 1
            elif str(doc) in val_docs:
                val_mask[id] = 1
                representation_mask[id] = 0
            else:
                not_in_either+=1
        
        print("\nNot_in_either = ", not_in_either)
        print("train_mask sum = ", sum(train_mask))
        print("val_mask sum = ", sum(val_mask))
        
        temp_dict = {}
        temp_dict['train_mask'] = list(train_mask)
        temp_dict['val_mask'] = list(val_mask)
        temp_dict['repr_mask'] = list(representation_mask)
        self.split_mask_file = os.path.join(self.comp_dir, dataset, self.split_mask_file)
        print("Writing split mask file in : ", self.split_mask_file)
        with open(self.split_mask_file, 'w+') as j:
            json.dump(temp_dict, j) 