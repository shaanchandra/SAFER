import json, time, random, datetime
import os, sys, glob, csv, re, argparse
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
from scipy.sparse import csr_matrix, lil_matrix, save_npz, load_npz
from collections import defaultdict
import seaborn as sns
import matplotlib.pylab as plt

import nltk
# nltk.download('punkt')
sys.path.append("..")
import torch



class GCN_PreProcess():
    def __init__(self, config, gossipcop=False, politifact=False, pheme=False):
        self.config = config
        self.data_dir = config['data_dir']
        self.datasets = []
        if politifact:
            self.datasets.append('politifact')
        if gossipcop:
            self.datasets.append('gossipcop')
        if pheme:
            self.datasets.append('pheme')
        
        if config['create_aggregate_folder']:
            self.create_aggregate_folder()
        if config['create_doc_user_folds']:
            self.create_doc_user_folds()
        if config['create_dicts']:
            self.create_dicts()
        if config['create_adj_matrix']:
            self.create_adj_matrix()
        if config['create_feat_matrix']:
            self.create_feat_matrix()
        if config['create_labels']:
            self.create_labels()
        if config['create_split_masks']:
            self.create_split_masks()
            
        # self.check_overlapping_users()
        # self.generate_graph_stats()
        # self.create_filtered_follower_following()
        
    def create_aggregate_folder(self):
        print("\n" + "-"*70 + "\n \t\t Creating aggregate files for PHEME dataset\n" + '-'*70)
        src_dir = os.path.join(os.getcwd(), '..', 'data', 'base_data', 'pheme_cv')
        events = ['ch', 'ebola', 'ferg', 'german', 'gurlitt', 'ottawa', 'putin', 'sydney', 'toronto']
        c=0
        # cmmnt_users = defaultdict(list)
        src_users = {}
        all_docs= []
        for fold, event in enumerate(events):
            print("\nIterating over {}...".format(event))
            data_dir = os.path.join(src_dir, event)
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if file.startswith('.') or file.startswith('structure') or file.startswith('annotation'):
                        continue
                    else:
                        src_file_path = os.path.join(root, file)
                        src_file = json.load(open(src_file_path, 'r'))
                        doc = file.split('.')[0]
                        user_id = str(src_file['user']['id'])
                        all_docs.append(doc)
                        src_users[doc] = user_id
                        c+=1
                        if c%2000 == 0:
                            print("{} done...".format(c))
                           
        print("\nTotal tweets/re-tweets in the data set = ", c)
        print("\nWriting all the info in the dir..")
        for doc in all_docs:
            temp_dict = {}
            temp_dict['source_user'] = src_users[doc]
                    
            write_file = os.path.join(os.getcwd(), '..', 'data', 'complete_data', 'pheme_cv', 'complete_mtl', '{}.json'.format(doc))
            with open(write_file, 'w+') as j:               
                json.dump(temp_dict, j)
        print("\nDONE..!!")
                
                
                
        
    def create_doc_user_folds(self):
        print("\n\n" + "-"*100 + "\n \t\t\t   Creating doc and user folds for  PHEME dataset \n" + '-'*100)
        all_users = dict()
        all_docs = dict()
        uniq_users = set()
        pheme_dir = os.path.join(os.getcwd(), '..', 'data', 'base_data', 'pheme_cv')
        events = ['ch', 'ebola', 'ferg', 'german', 'gurlitt', 'ottawa', 'putin', 'sydney', 'toronto']
        
        for fold, event in enumerate(events):
            c=0
            all_docs[event] = {}
            
            print("\nPreparing subset of stance docs...")
            splits = ['train', 'dev', 'test']
            subset_stance = []
            c=0
            for split in splits:
                stance_labels_file = os.path.join(os.getcwd(), '..', 'rumor_eval', 'rumoreval_crawl', '{}-key.json'.format(split))
                stance_labels = json.load(open(stance_labels_file, 'r'))['subtaskaenglish']
                for doc, annotation in stance_labels.items():
                    if str(doc).isdigit():
                        subset_stance.append(str(doc))
            
            subset_stance_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'subset_stance.json')
            print("subset_stance = ", len(subset_stance))
            print("Saving doc_splits dict in :", subset_stance_file)
            temp_dict = {'subset_stance' : subset_stance}
            with open(subset_stance_file, 'w+') as j:
                json.dump(temp_dict, j)
                
                
            
            event_users, event_veracity_docs, event_stance_docs, all_event_docs = set(), set(), set(), set()
            src_dir = os.path.join(pheme_dir, event)
            for root, dirs, files in os.walk(src_dir):
                for file in files:
                    if file.startswith('.') or file.startswith('structure') or file.startswith('annotation'):
                        continue
                    else:
                        c+=1
                        src_file_path = os.path.join(root, file)
                        src_tweet = json.load(open(src_file_path, 'r'))
                        doc = file.split('.')[0]
                        user_id = src_tweet['user']['id']
                        
                        if root.endswith('reactions') and str(doc) in subset_stance:
                            event_stance_docs.update([str(doc)])
                            uniq_users.update([str(user_id)])
                            event_users.update([str(user_id)])
                            all_event_docs.update([str(doc)])
                        elif root.endswith('source-tweets'):
                            event_veracity_docs.update([str(doc)])
                            uniq_users.update([str(user_id)])
                            event_users.update([str(user_id)])
                            all_event_docs.update([str(doc)])
            
            event_stance_docs = list(event_stance_docs)
            event_stance_docs = [e for e in list(event_stance_docs) if e not in list(event_veracity_docs)]
            all_users[event] = list(event_users)
            all_docs[event]['all'] = list(all_event_docs)
            all_docs[event]['stance'] = list(event_stance_docs)
            all_docs[event]['veracity'] = list(event_veracity_docs)
            print("\nFold {} done:".format(fold+1))
            print("Total docs in event {} = {}".format(event, len(all_event_docs)))
            print("Total veracity docs in event {} = {}".format(event, len(event_veracity_docs)))
            print("Total stance docs in event {} = {}".format(event, len(event_stance_docs)))
            print("Total users in event {} = {}".format(event, len(event_users)))
                        
        print("unique users = ", len(uniq_users))
        fold_user_file = os.path.join(self.data_dir, 'fold_users_mtl.json')
        print("\nSaving MTL fold users in  :", fold_user_file)
        with open(fold_user_file, 'w+') as j:
            json.dump(all_users, j)
        
        fold_doc_file = os.path.join(self.data_dir, 'fold_docs_mtl.json')
        print("Saving MTL fold docs in  :", fold_doc_file)
        with open(fold_doc_file, 'w+') as j:
            json.dump(all_docs, j)
        
        return None
    
    
    
    
    
    def create_dicts(self):
        
        events = ['ch', 'ebola', 'ferg', 'german', 'gurlitt', 'ottawa', 'putin', 'sydney', 'toronto']
        for fold, event in enumerate(events):
            print("\n\n" + "-"*100 + "\n \t\t\t   Creating dicts for  PHEME dataset fold {} \n".format(fold+1) + '-'*100)
            
            fold_user_file = os.path.join(self.data_dir, 'fold_users_mtl.json')
            fold_doc_file = os.path.join(self.data_dir, 'fold_docs_mtl.json')
            
            fold_users = json.load(open(fold_user_file, 'r'))
            fold_docs = json.load(open(fold_doc_file, 'r'))
            
            train_users, test_users = set(),  set()
            train_docs_stance, train_docs_veracity, test_docs_stance, test_docs_veracity = set(), set(), set(), set()
            for e in events:
                if e != event: 
                    train_users.update(fold_users[e])
                    train_docs_stance.update(fold_docs[e]['stance'])
                    train_docs_veracity.update(fold_docs[e]['veracity'])
                else: 
                    test_users.update(fold_users[e])
                    test_docs_stance.update(fold_docs[e]['stance'])
                    test_docs_veracity.update(fold_docs[e]['veracity'])
               
            # print("Common in stance and veracity = ", len(set(train_docs_stance).intersection(set(train_docs_veracity))))
            # print(set(train_docs_stance).intersection(set(train_docs_veracity)))
            # sys.exit()
            train_docs_stance, train_docs_veracity, train_users = list(train_docs_stance), list(train_docs_veracity), list(train_users)
            test_docs_stance, test_docs_veracity, test_users = list(test_docs_stance), list(test_docs_veracity), list(test_users)
            print("\nTotal train users = ", len(train_users))
            print("Total train docs = ", len(train_docs_stance + train_docs_veracity))
            print("Total stance train docs = ", len(train_docs_stance))
            print("Total veracity train docs = ", len(train_docs_veracity))
            print("Total test users = ", len(test_users))
            print("Total test docs = ", len(test_docs_stance) + len(test_docs_veracity))
            print("Total stance test docs = ", len(test_docs_stance))
            print("Total veracity test docs = ", len(test_docs_veracity))
            
            temp_dict = {}
            temp_dict['train_docs_stance'] = train_docs_stance
            temp_dict['train_docs_veracity'] = train_docs_veracity
            temp_dict['test_docs_stance'] = test_docs_stance
            temp_dict['test_docs_veracity'] = test_docs_veracity
            doc_splits_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'doc_splits_mtl.json')
            print("Saving doc_splits dict in :", doc_splits_file)
            with open(doc_splits_file, 'w+') as j:
                json.dump(temp_dict, j)
            
                      
            print("\nPreparing doc2id and user2id...")
            doc2id_train = {}
            node_type = []
            for train_count_stance, doc in enumerate(train_docs_stance):
                doc2id_train[str(doc)] = train_count_stance
                node_type.append(1)
            print("doc2id_train (stance) = ", len(doc2id_train))
            
            orig_len = len(doc2id_train)
            for train_count_veracity, doc in enumerate(train_docs_veracity):
                doc2id_train[str(doc)] = train_count_veracity + orig_len
                node_type.append(2)
            print("doc2id_train (veracity) = ", len(doc2id_train))


            # for val_count, doc in enumerate(val_docs):
            #     doc2id_train[str(doc)] = val_count + len(train_docs)
            # print("doc2id_train = ", len(doc2id_train))
            # doc2id_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'doc2id_train.json')
            # print("Saving doc2id dict in :", doc2id_file)
            # with open(doc2id_file, 'w+') as j:
            #     json.dump(doc2id_train, j)
                            
            
            a = set(train_users)
            b = set(test_users)
            print("\nUsers common between train and test = ", len(a.intersection(b)))

            all_users = list(set(train_users + test_users))                

            
            user2id_train = {}
            for count, user in enumerate(all_users):
                user2id_train[str(user)] = count + len(doc2id_train)
                node_type.append(3)
            user2id_train_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'user2id_train_mtl.json')
            print("user2id = ", len(user2id_train))
            print("Saving user2id_train in : ", user2id_train_file)
            with open(user2id_train_file, 'w+') as j:
                json.dump(user2id_train, j)
            
            
            node2id = doc2id_train.copy()
            node2id.update(user2id_train)
            print("node2id size = ", len(node2id))
            node2id_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'node2id_lr_train_mtl.json')
            print("Saving node2id_lr_train in : ", node2id_file)
            with open(node2id_file, 'w+') as json_file:
                json.dump(node2id, json_file)
                
            node_type_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'node_type_lr_train_mtl.npy')
            node_type = np.array(node_type)
            print(node_type.shape)
            print("Saving node_type in :", node_type_file)
            np.save(node_type_file, node_type, allow_pickle=True)
            
            print("\nAdding test docs..")
            orig_doc2id_len = len(doc2id_train)
            for test_count_stance, doc in enumerate(test_docs_stance):
                doc2id_train[str(doc)] = test_count_stance + len(node2id)
            print("doc2id_train = ", len(doc2id_train))
            
            orig_doc2id_len = len(doc2id_train)
            for test_count_veracity, doc in enumerate(test_docs_veracity):
                doc2id_train[str(doc)] = orig_doc2id_len + test_count_veracity
            print("doc2id_train = ", len(doc2id_train))
            
            doc2id_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'doc2id_train_mtl.json')
            print("Saving doc2id dict in :", doc2id_file)
            with open(doc2id_file, 'w+') as j:
                json.dump(doc2id_train, j)
            
            node2id = doc2id_train.copy()
            node2id.update(user2id_train)
            print("node2id size = ", len(node2id))
            node2id_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'node2id_lr_mtl.json')
            print("Saving node2id_lr_train in : ", node2id_file)
            with open(node2id_file, 'w+') as json_file:
                json.dump(node2id, json_file)

            print("Done ! All files written..")
                    
        return None
                    
                    
                    
                    
                    
    def create_adj_matrix(self):
        # Commenters are connected to the source-tweet
        
        events = ['ch', 'ebola', 'ferg', 'german', 'gurlitt', 'ottawa', 'putin', 'sydney', 'toronto']
        
        for fold, event in enumerate(events):
            print("\n\n" + "-"*100 + "\n \t\t\tAnalyzing  PHEME dataset and fold {} for adj_matrix\n".format(fold+1) + '-'*100)
            
            user2id_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'user2id_train_mtl.json')
            doc2id_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'doc2id_train_mtl.json')
            
            user2id = json.load(open(user2id_file, 'r'))
            doc2id = json.load(open(doc2id_file, 'r'))
                               
            doc_splits_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'doc_splits_mtl.json')
            doc_splits = json.load(open(doc_splits_file, 'r'))
            test_docs = doc_splits['test_docs_stance'] + doc_splits['test_docs_veracity']
            train_docs = doc_splits['train_docs_stance']  + doc_splits['train_docs_veracity']
                            
            num_users, num_docs = len(user2id), len(train_docs)+len(test_docs)
            print("\nNo.of unique users = ", num_users)
            print("No.of docs = ", num_docs)
            
            # Creating the adjacency matrix (doc-user edges)
            adj_matrix = lil_matrix((num_docs+num_users, num_users+num_docs))
            edge_type = lil_matrix((num_docs+num_users, num_users+num_docs))
            # adj_matrix = np.zeros((num_docs+num_users, num_users+num_docs))
            # adj_matrix_file = './data/complete_data/adj_matrix_pheme.npz'
            # adj_matrix = load_npz(adj_matrix_file)
            # adj_matrix = lil_matrix(adj_matrix)
            # Creating self-loops for each node (diagonals are 1's)
            for i in range(adj_matrix.shape[0]):
                adj_matrix[i,i] = 1
                edge_type[i,i] = 1
            print_iter = int(num_docs/10)
            print("\nSize of adjacency matrix = {} \nPrinting every  {} docs".format(adj_matrix.shape, print_iter))
            start = time.time()
            
            print("\nPreparing entries for doc-user pairs...")
            # with open('./data/complete_data/doc2tags_{}.json'.format(dataset),'r') as j:
            #    doc2tags = json.load(j)
            src_dir = os.path.join(self.data_dir, 'complete_mtl')
            for root, dirs, files in os.walk(src_dir):
                for count, file in enumerate(files):
                    src_file_path = os.path.join(root, file)
                    doc_key = file.split(".")[0]
                    src_file = json.load(open(src_file_path, 'r'))
                    if str(doc_key) in doc2id and str(src_file['source_user']) in user2id:
                        adj_matrix[doc2id[str(doc_key)], user2id[str(src_file['source_user'])]] = 1
                        adj_matrix[user2id[str(src_file['source_user'])], doc2id[str(doc_key)]] = 1
                        edge_type[doc2id[str(doc_key)], user2id[str(src_file['source_user'])]] = 2
                        edge_type[user2id[str(src_file['source_user'])], doc2id[str(doc_key)]] = 2
                            
                if count%print_iter==0:
                    print("{} / {} done..".format(count+1, num_docs))


            end = time.time() 
            hrs, mins, secs = self.calc_elapsed_time(start, end)
            print("Done. Took {}hrs and {}mins and {}secs\n".format(hrs, mins, secs))  
            print("Non-zero entries = ", adj_matrix.count_nonzero())
            print("Non-zero entries edge_type = ", edge_type.count_nonzero())
            # print("Non-zero entries = ", len(np.nonzero(adj_matrix)[0]))
            
            
            print("\nPreparing entries for doc-comment pairs...")
            # with open('./data/complete_data/doc2tags_{}.json'.format(dataset),'r') as j:
            #    doc2tags = json.load(j)
            pheme_dir = os.path.join(os.getcwd(), '..',  'data', 'base_data', 'pheme_cv')
            src_dir = os.path.join(pheme_dir, event)
            for root, dirs, files in os.walk(src_dir):
                for file in files:
                    if file.startswith('.') or file.startswith('structure') or file.startswith('annotation'):
                        continue
                    else:
                        src_tweet = root.split('\\')[-2]
                        comment =  file.split('.')[0]
                        
                        if str(src_tweet) in doc2id and str(comment) in doc2id and src_tweet!=comment:
                            adj_matrix[doc2id[str(src_tweet)], doc2id[str(comment)]] = 1
                            adj_matrix[doc2id[str(comment)], doc2id[str(src_tweet)]] = 1
                            edge_type[doc2id[str(src_tweet)], doc2id[str(comment)]] = 3
                            edge_type[doc2id[str(comment)], doc2id[str(src_tweet)]] = 3
                            
                if count%print_iter==0:
                    print("{} / {} done..".format(count+1, num_docs))


            end = time.time() 
            hrs, mins, secs = self.calc_elapsed_time(start, end)
            print("Done. Took {}hrs and {}mins and {}secs\n".format(hrs, mins, secs))  
            print("Non-zero entries = ", adj_matrix.count_nonzero())
            print("Non-zero entries edge_type = ", edge_type.count_nonzero())
            
            
            
            # Creating the adjacency matrix (user-user edges) - follower/ing
            user_contexts = ['user_followers_filtered', 'user_following_filtered']
            start = time.time()
            key_errors, not_found, overlaps = 0,0,0
            print("\nPreparing entries for user-user pairs...")
            print_iter = int(num_users/10)
            print("Printing every {}  users done".format(print_iter))
            
            for user_context in user_contexts:
                print("    - from {}  folder...".format(user_context))
                src_dir2 = os.path.join(self.data_dir, user_context)
                for root, dirs, files in os.walk(src_dir2):
                    for count, file in enumerate(files):
                        src_file_path = os.path.join(root, file)
                        # user_id = src_file_path.split(".")[0]
                        src_file = json.load(open(src_file_path, 'r'))
                        user_id = int(src_file['user_id'])
                        if str(user_id) in user2id:
                            followers = src_file['followers'] if user_context == 'user_followers_filtered' else src_file['following']   
                            followers = list(map(int, followers))
                            for follower in followers:
                                if str(follower) in user2id:
                                    adj_matrix[user2id[str(user_id)], user2id[str(follower)]]=1
                                    adj_matrix[user2id[str(follower)], user2id[str(user_id)]]=1
                                    edge_type[user2id[str(follower)], user2id[str(user_id)]] = 4
                                    edge_type[user2id[str(user_id)], user2id[str(follower)]] = 4
                                    
                        else:
                            not_found +=1
                        # if count%print_iter==0:
                        #     # print("{}/{} done..  Non-zeros =  {}".format(count+1, num_users, adj_matrix.getnnz()))
                        #     print("{}/{} done..  Non-zeros =  {}".format(count+1, num_users, len(np.nonzero(adj_matrix)[0])))
                                         
            hrs, mins, secs = self.calc_elapsed_time(start, time.time())
            print("Done. Took {}hrs and {}mins and {}secs\n".format(hrs, mins, secs))
            print("Not found user_ids = ", not_found)
            print("Total Non-zero entries = ", adj_matrix.getnnz())
            # print("Total Non-zero entries = ", len(np.nonzero(adj_matrix)[0]))
            
            filename = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'adj_matrix_lr_mtl.npz')
            # filename = self.data_dir+ '/complete_data' + '/adj_matrix_{}.npy'.format(dataset)
            print("\nMatrix construction done! Saving in  {}".format(filename))
            save_npz(filename, adj_matrix.tocsr())
            # np.save(filename, adj_matrix)
            
            filename = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'edge_type_lr_mtl.npz')
            print("\nEdge type saving in  {}".format(filename))
            save_npz(filename, edge_type.tocsr())
            
            # Creating an edge_list matrix of adj_matrix as required by some GCN frameworks
            print("\nCreating edge_index format of adj_matrix...")
            start = time.time()
            rows, cols = adj_matrix.nonzero()
            
            edge_index = np.vstack((np.array(rows), np.array(cols)))
            print("Edge index shape = ", edge_index.shape)
            
            edge_matrix_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'adj_matrix_lr_edge_mtl.npy')
            print("saving edge_list format in :  ", edge_matrix_file)
            np.save(edge_matrix_file, edge_index, allow_pickle=True)
            
            # rows, cols = edge_type.nonzero()
            # edge_index = np.vstack((np.array(rows), np.array(cols)))
            edge_index = edge_type[edge_type.nonzero()]
            edge_index = edge_index.toarray()
            edge_index = edge_index.squeeze(0)
            print("edge_type shape = ", edge_index.shape)
            edge_matrix_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'edge_type_lr_edge_mtl.npy')
            print("saving edge_type edge list format in :  ", edge_matrix_file)
            np.save(edge_matrix_file, edge_index, allow_pickle=True)
            
            
            hrs, mins, secs = self.calc_elapsed_time(start, time.time())
            print("Done. Took {}hrs and {}mins and {}secs\n".format(hrs, mins, secs))
        return None
    
    
    
    
    
    def create_feat_matrix(self, binary=True):
        events = ['ch', 'ebola', 'ferg', 'german', 'gurlitt', 'ottawa', 'putin', 'sydney', 'toronto']
        
        for fold, event in enumerate(events):
            print("\n\n" + "-"*100 + "\n \t\t\tAnalyzing  PHEME dataset and fold {} for feat_matrix\n".format(fold+1) + '-'*100)
            
            user2id_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'user2id_train_mtl.json')
            doc2id_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'doc2id_train_mtl.json')
            
            user2id = json.load(open(user2id_file, 'r'))
            doc2id = json.load(open(doc2id_file, 'r'))
                               
            doc_splits_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'doc_splits_mtl.json')
            doc_splits = json.load(open(doc_splits_file, 'r'))
            test_docs = doc_splits['test_docs_stance'] + doc_splits['test_docs_veracity']
            train_docs = doc_splits['train_docs_stance']  + doc_splits['train_docs_veracity']
                            
            num_users, num_docs = len(user2id), len(train_docs)+len(test_docs)
            print("\nNo.of unique users = ", num_users)
            print("No.of docs = ", num_docs)
  
            N = len(train_docs) + len(test_docs) + len(user2id)
            
            vocab = {}
            vocab_size=0
            start = time.time()
            print("\nBuilding vocabulary...") 
            vocab_size=0
            c=0
            not_considered=0
            src_doc_dir = os.path.join(os.getcwd(), '..', 'data', 'base_data', 'pheme_cv')
            for root, dirs, files in os.walk(src_doc_dir):
                for file in files:
                    if file.startswith('.') or file.startswith('structure') or file.startswith('annotation'):
                        continue
                    doc = file.split('.')[0]
                    if str(doc) not in test_docs and str(doc) in doc2id:
                        src_file_path = os.path.join(root, file)
                        src_tweet = json.load(open(src_file_path, 'r'))
                        text = src_tweet['text']
                        text = re.sub(r'#[\w-]+', 'hashtag', text)
                        text = re.sub(r'@[\w-]+', 'mention', text)
                        text = re.sub(r'https?://\S+', 'url', text)
                        text = text.replace('\n', ' ')
                        text = text.replace('\t', ' ')
                        # text = re.sub(r"[^A-Za-z(),!?\'`]", " ", text)
                        text = nltk.word_tokenize(text)
                        for token in text:
                            if token not in vocab.keys():
                                vocab[token] = vocab_size
                                vocab_size+=1  
                    else:
                        not_considered+=1
    
            hrs, mins, secs = self.calc_elapsed_time(start, time.time())
            print("Done. Took {}hrs and {}mins and {}secs\n".format(hrs, mins, secs))
            print("not_considered = ", not_considered)
            print("Size of vocab =  ", vocab_size)
            vocab_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'vocab_mtl.json')
            print("Saving vocab for fold {}  at:  {}".format(fold+1, vocab_file))
            with open(vocab_file, 'w+') as v:
                json.dump(vocab, v)
            
            
            feat_matrix = lil_matrix((N, vocab_size))
            print("\nSize of feature matrix = ", feat_matrix.shape)
            print("\nCreating feat_matrix entries for docs nodes...")
            start = time.time()
            # split_docs = train_docs+ val_docs
            # split_users = train_users + val_users
                
            c=0
            src_doc_dir = os.path.join(os.getcwd(), '..', 'data', 'base_data', 'pheme_cv')
            for root, dirs, files in os.walk(src_doc_dir):
                for file in files:
                    if file.startswith('.') or file.startswith('structure') or file.startswith('annotation'):
                        continue
                    else:
                        doc = file.split('.')[0]
                        if str(doc) in doc2id:
                            # feat_matrix[doc2id[str(doc)], :] = np.random.random(len(vocab)) > 0.99
                            
                            src_file_path = os.path.join(root, file)
                            src_tweet = json.load(open(src_file_path, 'r'))
                            text = src_tweet['text']
                            text = re.sub(r'#[\w-]+', 'hashtag', text)
                            text = re.sub(r'@[\w-]+', 'mention', text)
                            text = re.sub(r'https?://\S+', 'url', text)
                            text = text.replace('\n', ' ')
                            text = text.replace('\t', ' ')
                            # text = re.sub(r"[^A-Za-z(),!?\'`]", " ", text)
                            text = nltk.word_tokenize(text)
                            for token in text:
                                if token in vocab.keys():
                                    feat_matrix[doc2id[str(doc)], vocab[token]] = 1
                            c+=1
                            if c%5000 == 0:
                                print("{} done...".format(c))
                
            hrs, mins, secs = self.calc_elapsed_time(start, time.time())
            print("Done. Took {}hrs and {}mins and {}secs\n".format(hrs, mins, secs))
            
            sum_1 = np.array(feat_matrix.sum(axis=1)).squeeze(1)
            print(sum_1.shape)
            idx = np.where(sum_1==0)
            print(len(idx[0]))
            
            
            print("\nCreating feat_matrix entries for users nodes...")
            start = time.time()
            not_found, use = 0,0
            # user_splits = json.load(open('./data/complete_data/{}/user_splits.json'.format(dataset), 'r'))
            # train_users = user_splits['train_users']
            src_dir = os.path.join(self.data_dir,'complete_mtl')
            user_contexts = ['user_followers_filtered', 'user_following_filtered']
            for root, dirs, files in os.walk(src_dir):
                for count, file in enumerate(files):
                    print_iter = int(len(files) / 5)
                    src_file_path = os.path.join(root, file)
                    src_file = json.load(open (src_file_path, 'r'))
                    user = src_file['source_user']
                    doc_key = file.split(".")[0]
                    # if str(doc_key) in train_docs:
                    # Each user of this doc has its features as the features of the doc
                    if str(doc_key) in doc2id and str(user) in user2id:
                        feat_matrix[user2id[str(user)], :] += feat_matrix[doc2id[str(doc_key)], :]
                    
                            
                    if count%print_iter==0:
                        print(" {} / {} done..".format(count+1, len(files)))
                        # print(datetime.datetime.now())
                    
            hrs, mins, secs = self.calc_elapsed_time(start, time.time())
            print(not_found, use)
            print("Done. Took {}hrs and {}mins and {}secs\n".format(hrs, mins, secs))
            
            feat_matrix = feat_matrix >= 1
            feat_matrix = feat_matrix.astype(int)
            
            # Sanity Checks
            sum_1 = np.array(feat_matrix.sum(axis=1)).squeeze(1)
            print(sum_1.shape)
            idx = np.where(sum_1==0)
            print(len(idx[0]))
            
            filename = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'feat_matrix_lr_mtl.npz')
            print("Matrix construction done! Saving in :   {}".format(filename))
            save_npz(filename, feat_matrix.tocsr())
            
            
            
    def convert_annotations(self, annotation, string = False):
        if 'misinformation' in annotation.keys() and 'true'in annotation.keys():
            if int(annotation['misinformation'])==0 and int(annotation['true'])==0:
                if string:
                    label = "unverified"
                else:
                    label = 2
            elif int(annotation['misinformation'])==0 and int(annotation['true'])==1 :
                if string:
                    label = "true"
                else:
                    label = 1
            elif int(annotation['misinformation'])==1 and int(annotation['true'])==0 :
                if string:
                    label = "false"
                else:
                    label = 0
            elif int(annotation['misinformation'])==1 and int(annotation['true'])==1:
                print ("OMG! They both are 1!")
                print(annotation['misinformation'])
                print(annotation['true'])
                label = None
                
        elif 'misinformation' in annotation.keys() and 'true' not in annotation.keys():
            # all instances have misinfo label but don't have true label
            if int(annotation['misinformation'])==0:
                if string:
                    label = "unverified"
                else:
                    label = 2
            elif int(annotation['misinformation'])==1:
                if string:
                    label = "false"
                else:
                    label = 0
                    
        elif 'true' in annotation.keys() and 'misinformation' not in annotation.keys():
            print ('Has true not misinformation')
            label = None
        else:
            print('No annotations')
            label = None              
        return label
    
    
    def convert_stance_annotations(self, annotation):
        if annotation == 'support':
            label = 0
        elif annotation == 'deny':
            label = 1
        elif annotation == 'query':
            label = 2
        elif annotation == 'comment':
            label = 3
        
        return label
            
      
    """
    Create labels for each node of the graph
    """
    def create_labels(self):
        events = ['ch', 'ebola', 'ferg', 'german', 'gurlitt', 'ottawa', 'putin', 'sydney', 'toronto']
        
        for fold, event in enumerate(events):
            print("\n\n" + "-"*100 + "\n \t\t   Creating labels for fold {}\n".format(fold+1) + '-'*100)
            
            user2id_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'user2id_train_mtl.json')
            doc2id_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'doc2id_train_mtl.json')
            fold_user_file = os.path.join(self.data_dir, 'fold_users_mtl.json')
            fold_doc_file = os.path.join(self.data_dir, 'fold_docs_mtl.json')

            fold_users = json.load(open(fold_user_file, 'r'))
            fold_docs = json.load(open(fold_doc_file, 'r'))
            user2id = json.load(open(user2id_file, 'r'))
            doc2id = json.load(open(doc2id_file, 'r'))
                               
            doc_splits_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'doc_splits_mtl.json')
            doc_splits = json.load(open(doc_splits_file, 'r'))
            adj_matrix_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'adj_matrix_lr_mtl.npz')
            adj_matrix = load_npz(adj_matrix_file)
            N,_ = adj_matrix.shape
            del adj_matrix

            stance_docs = fold_docs[event]['stance']
            veracity_docs = fold_docs[event]['veracity']
                
            print("\nCreating doc2labels dictionary...")
            print("--preparing veracity labels...")
            doc2labels_veracity = {}
            c=0
            pheme_dir = './data/base_data/pheme_cv'
            for root, dirs, files in os.walk(pheme_dir):
                for file in files:
                    if not file.startswith('annotation'):
                        continue
                    src_file_path = os.path.join(root, file)
                    doc = root.split('\\')[-1]
                    with open(src_file_path, 'r') as j:
                        annotation = json.load(j)
                        doc2labels_veracity[str(doc)] = self.convert_annotations(annotation, string = False)
                        c+=1
                        if c%500 == 0:
                            print("{} done..".format(c))
            
            
            veracity_labels = list(doc2labels_veracity.values())
            t,f,u = self.get_label_distribution_veracity(veracity_labels)
            print("t, f, u = {}, {}, {}".format(t,f,u))
            
            # print(len(doc2labels.keys()))
            # print(len(doc2id.keys()) - len(doc_splits['test_docs']))
            # assert len(doc2labels.keys()) == len(doc2id.keys()) - len(doc_splits['test_docs'])
            print("Len of doc2labels = {}\n".format(len(doc2labels_veracity)))
            labels_dict_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'doc2labels_veracity_mtl.json')
            print("Saving labels_dict for fold {} at:  {}".format(fold+1, labels_dict_file))
            with open(labels_dict_file, 'w+') as v:
                json.dump(doc2labels_veracity, v)
            
            labels_list = np.zeros(N)
            for key,value in doc2labels_veracity.items():
                labels_list[doc2id[str(key)]] = value
                          
            # Sanity Checks
            # print(sum(labels_list))
            # print(len(labels_list))
            # print(sum(labels_list[2402:]))
            # print(sum(labels_list[:2402]))
            
            filename = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'labels_list_veracity_mtl.json')
            temp_dict = {}
            temp_dict['labels_list'] = list(labels_list)
            print("Labels list construction done! Saving in :   {}".format(filename))
            with open(filename, 'w+') as v:
                json.dump(temp_dict, v)
            
            
            # Create the all_labels file
            all_labels = np.zeros(N)
            all_labels_data_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'all_labels_veracity_mtl.json')
            for doc in doc2labels_veracity.keys():
                all_labels[doc2id[str(doc)]] = doc2labels_veracity[str(doc)]
            
            temp_dict = {}
            temp_dict['all_labels'] = list(all_labels)
            print("Sum of all_labels this test set = ", sum(all_labels))
            print("Len of labels = ", len(all_labels))
            with open(all_labels_data_file, 'w+') as j:
                json.dump(temp_dict, j)
                
            ##################################################################################################
                
                
            print("\n--preparing stance labels...")
            splits = ['train', 'dev', 'test']
            doc2labels_stance = {}
            c=0
            for split in splits:
                stance_labels_file = os.path.join(os.getcwd(), '..', 'rumor_eval', 'rumoreval_crawl', '{}-key.json'.format(split))
                stance_labels = json.load(open(stance_labels_file, 'r'))['subtaskaenglish']
                for doc, annotation in stance_labels.items():
                    if str(doc) in doc2id:
                        c+=1
                        doc2labels_stance[str(doc)] = self.convert_stance_annotations(annotation)
            
            stance_labels = list(doc2labels_stance.values())
            s,d,q,c = self.get_label_distribution_stance(stance_labels)
            print("s, d, q, c = {}, {}, {}, {}".format(s,d,q,c))
            
            
            # print(len(doc2labels.keys()))
            # print(len(doc2id.keys()) - len(doc_splits['test_docs']))
            # assert len(doc2labels.keys()) == len(doc2id.keys()) - len(doc_splits['test_docs'])
            print("Len of doc2labels = {}\n".format(len(doc2labels_stance)))
            print("Total stance labels found = ", c)
            labels_dict_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'doc2labels_stance_mtl.json')
            print("Saving labels_dict for fold {} at:  {}".format(fold+1, labels_dict_file))
            with open(labels_dict_file, 'w+') as v:
                json.dump(doc2labels_stance, v)
                
            
            labels_list = np.zeros(N)
            for key,value in doc2labels_stance.items():
                labels_list[doc2id[str(key)]] = value
                          
            # Sanity Checks
            # print(sum(labels_list))
            # print(len(labels_list))
            # print(sum(labels_list[2402:]))
            # print(sum(labels_list[:2402]))
            
            filename = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'labels_list_stance_mtl.json')
            temp_dict = {}
            temp_dict['labels_list'] = list(labels_list)
            print("Labels list construction done! Saving in :   {}".format(filename))
            with open(filename, 'w+') as v:
                json.dump(temp_dict, v)
            
            
            # Create the all_labels file
            all_labels = np.zeros(N)
            all_labels_data_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'all_labels_stance_mtl.json')
            for doc in doc2labels_stance.keys():
                all_labels[doc2id[str(doc)]] = doc2labels_stance[str(doc)]
            
            temp_dict = {}
            temp_dict['all_labels'] = list(all_labels)
            print("Sum of all_labels this test set = ", sum(all_labels))
            print("Len of labels = ", len(all_labels))
            with open(all_labels_data_file, 'w+') as j:
                json.dump(temp_dict, j)
        return None
    
    
    
    def create_split_masks(self):
        # Create and save split masks
        events = ['ch', 'ebola', 'ferg', 'german', 'gurlitt', 'ottawa', 'putin', 'sydney', 'toronto']
        
        for fold, event in enumerate(events):
            print("\n\n" + "-"*100 + "\n \t\t   Creating Split masks for fold {}\n".format(fold+1) + '-'*100)
            
            user2id_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'user2id_train_mtl.json')
            doc2id_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'doc2id_train_mtl.json')
            split_mask_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'split_mask_mtl.json')
            adj_matrix_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'adj_matrix_lr_mtl.npz')
            doc_splits_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'doc_splits_mtl.json')
            doc2labels_stance_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'doc2labels_stance_mtl.json')
            fold_user_file = os.path.join(self.data_dir, 'fold_users_mtl.json')
            fold_doc_file = os.path.join(self.data_dir, 'fold_docs_mtl.json')

            fold_users = json.load(open(fold_user_file, 'r'))
            fold_docs = json.load(open(fold_doc_file, 'r'))
            doc2labels_stance = json.load(open(doc2labels_stance_file, 'r'))
            
            user2id = json.load(open(user2id_file, 'r'))
            doc2id = json.load(open(doc2id_file, 'r'))
            adj = load_npz(adj_matrix_file)
            N, _ = adj.shape
            del adj
                               
            doc_splits = json.load(open(doc_splits_file, 'r'))
            test_docs_stance, test_docs_veracity = doc_splits['test_docs_stance'] , doc_splits['test_docs_veracity']
            train_docs_stance, train_docs_veracity = doc_splits['train_docs_stance'] , doc_splits['train_docs_veracity']

            train_mask_stance, train_mask_veracity, test_mask_stance, test_mask_veracity = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N) # np.zeros(test_n)
            representation_mask_stance, representation_mask_veracity = np.ones(N), np.ones(N)

            not_in_either=0
            train_labels, test_labels = [], []
            train_stance_labels, test_stance_labels = [], []
            for doc, id in doc2id.items():
                if doc in train_docs_stance and doc in doc2labels_stance:
                    train_mask_stance[id] = 1
                    train_stance_labels.append(doc2labels_stance[str(doc)])
                if doc in train_docs_veracity:
                    train_mask_veracity[id] = 1
                if doc in test_docs_stance and doc in doc2labels_stance:
                    test_mask_stance[id] = 1
                    representation_mask_stance[id] = 0
                    test_stance_labels.append(doc2labels_stance[str(doc)])
                if doc in test_docs_veracity:
                    test_mask_veracity[id] = 1
                    representation_mask_veracity[id] = 0
                else:
                    not_in_either+=1
            
            print("\nnot_in_either = ", not_in_either)
            print("train_mask_stance sum = ", sum(train_mask_stance))
            print("train_mask_veracity sum = ", sum(train_mask_veracity))
            print("test_mask_stance sum = ", sum(test_mask_stance))
            print("test_mask_veracity sum = ", sum(test_mask_veracity))
            print("representation_mask_stance sum = ", sum(representation_mask_stance))
            print("representation_mask_veracity sum = ", sum(representation_mask_veracity))
            print(event)
            s,d,q,c = self.get_label_distribution_stance(train_stance_labels)
            print("s, d, q, c = {}, {}, {}, {}".format(s,d,q,c))
            
            s,d,q,c = self.get_label_distribution_stance(test_stance_labels)
            print("s, d, q, c = {}, {}, {}, {}".format(s,d,q,c))
            
            
            temp_dict = {}
            temp_dict['train_mask_stance'] = list(train_mask_stance)
            temp_dict['train_mask_veracity'] = list(train_mask_veracity)
            temp_dict['test_mask_stance'] = list(test_mask_stance)
            temp_dict['test_mask_veracity'] = list(test_mask_veracity)
            temp_dict['representation_mask_stance'] = list(representation_mask_stance)
            temp_dict['representation_mask_veracity'] = list(representation_mask_veracity)
            # with open(split_mask_file, 'w+') as j:
            #     json.dump(temp_dict, j)
            
                  
        return None
    
    
    
    def get_label_distribution_veracity(self, labels):  
        true = labels.count(1)
        false = labels.count(0)
        unverified = labels.count(2)
        denom = true + false + unverified    
        return true/denom, false/denom, unverified/denom
    
    
    def get_label_distribution_stance(self, labels):  
        s = labels.count(0)
        d = labels.count(1)
        q = labels.count(2)
        c = labels.count(3)
        print("& {} & {} & {} & {} &".format(s,d,q,c))
        denom = s+d+q+c
        if denom==0:
            denom=1
        return s/denom, d/denom, q/denom, c/denom

    def calc_elapsed_time(self, start, end):
        hours, rem = divmod(end-start, 3600)
        time_hours, time_rem = divmod(end, 3600)
        minutes, seconds = divmod(rem, 60)
        time_mins, _ = divmod(time_rem, 60)
        return int(hours), int(minutes), int(seconds)
                
        
                
        
      


if __name__== '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, default = '../data/complete_data/pheme_cv',
                          help='path to dataset folder that contains the folders to gossipcop or politifact folders (raw data)') 
    
    parser.add_argument('--create_aggregate_folder', type = bool, default = False,
                          help='Aggregate only user ids from different folders of tweets/retweets to a single place')
    parser.add_argument('--create_doc_user_folds', type = bool, default = False,
                          help='create dictionaries of user_ids and doc_ids present in different CV folds')
    parser.add_argument('--create_dicts', type = bool, default = False,
                          help='Create doc2id and node2id dictionaries')
    parser.add_argument('--create_adj_matrix', type = bool, default = False,
                          help='create adjacency matrix for a given dataset with commenting users connected to source tweet')
    parser.add_argument('--create_feat_matrix', type = bool, default = False,
                          help='To create feature matrix for a given dataset')
    parser.add_argument('--create_labels', type = bool, default = False,
                          help='To create labels for all the nodes')
    parser.add_argument('--create_split_masks', type = bool, default = True,
                          help='To create node masks for data splits')
    
    
    args, unparsed = parser.parse_known_args()
    config = args.__dict__
    
    preprocesser = GCN_PreProcess(config, pheme=True)