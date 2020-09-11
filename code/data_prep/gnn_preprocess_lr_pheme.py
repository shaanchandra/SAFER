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
    def __init__(self, config, pheme=True):
        self.config = config
        self.data_dir = config['data_dir']
        self.datasets = []
        
    
        if pheme:
            self.datasets.append('pheme')
        
        if config['create_aggregate_folder']:
            self.create_aggregate_folder()
        if config['create_doc_user_folds']:
            self.create_doc_user_folds()
        if config['create_dicts']:
            self.create_dicts()
        if config['create_adj_matrix_tweet']:
            self.create_adj_matrix_tweet()
        if config['create_adj_matrix_user']:
            self.create_adj_matrix_user()
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
        cmmnt_users = defaultdict(list)
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
                        doc = root.split('\\')[-2]
                        user_id = str(src_file['user']['id'])
                        all_docs.append(doc)
                        if root.endswith("source-tweets"):
                            src_users[doc] = user_id
                        else:
                            cmmnt_users[doc].append(user_id)
                        c+=1
                        if c%2000 == 0:
                            print("{} done...".format(c))
                           
        print("\nTotal tweets/re-tweets in the data set = ", c)
        print("\nWriting all the info in the dir..")
        for doc in all_docs:
            temp_dict = {}
            if doc in src_users:
                temp_dict['source_user'] = src_users[doc]
            if doc in cmmnt_users:
                temp_dict['users'] = list(set(cmmnt_users[doc]))
            else:
                temp_dict['users'] = []
                    
            write_file = './data/complete_data/pheme/complete/{}.json'.format(doc)
            with open(write_file, 'w+') as j:               
                json.dump(temp_dict, j)
        print("\nDONE..!!")
                
                
                
        
    def create_doc_user_folds(self):
        print("\n\n" + "-"*100 + "\n \t\t\t   Creating doc and user folds for  PHEME dataset \n" + '-'*100)
        
        all_users = dict()
        all_docs = dict()
        uniq_users = set()
        pheme_dir = '../data/base_data/pheme_cv'
        events = ['ch', 'ebola', 'ferg', 'german', 'gurlitt', 'ottawa', 'putin', 'sydney', 'toronto']
        
        for fold, event in enumerate(events):
            c=0
            event_users, event_docs = [], set()
            src_dir = os.path.join(pheme_dir, event)
            for root, dirs, files in os.walk(src_dir):
                for file in files:
                    if file.startswith('.') or file.startswith('structure') or file.startswith('annotation'):
                        continue
                    else:
                        c+=1
                        src_file_path = os.path.join(root, file)
                        src_tweet = json.load(open(src_file_path, 'r'))
                        doc = root.split('\\')[-2]
                        user_id = src_tweet['user']['id']
                        event_users.append(str(user_id))
                        event_docs.update([str(doc)])
                        uniq_users.update([str(user_id)])

            all_users[event] = event_users
            all_docs[event] = list(event_docs)
            print("\nFold {} done:".format(fold+1))
            print("Total docs in event {} = {}".format(event, len(event_docs)))
            print("Total users in event {} = {}".format(event, len(event_users)))
                        
        print("unique users = ", len(uniq_users))
        fold_user_file = os.path.join(self.data_dir, 'fold_users.json')
        print("\nSaving fold users in  :", fold_user_file)
        with open(fold_user_file, 'w+') as j:
            json.dump(all_users, j)
        
        fold_doc_file = os.path.join(self.data_dir, 'fold_docs.json')
        print("Saving fold docs in  :", fold_doc_file)
        with open(fold_doc_file, 'w+') as j:
            json.dump(all_docs, j)
        
        return None
    
    
    
    
    
    def create_dicts(self):
        
        events = ['ch', 'ebola', 'ferg', 'german', 'gurlitt', 'ottawa', 'putin', 'sydney', 'toronto']
        for fold, event in enumerate(events):
            print("\n\n" + "-"*100 + "\n \t\t\t   Creating dicts for  PHEME dataset fold {} \n".format(fold+1) + '-'*100)
            
            fold_user_file = os.path.join(self.data_dir, 'fold_users.json')
            fold_doc_file = os.path.join(self.data_dir, 'fold_docs.json')
            
            fold_users = json.load(open(fold_user_file, 'r'))
            fold_docs = json.load(open(fold_doc_file, 'r'))
            
            train_users, val_users, test_users = set(), set(), set()
            train_docs, val_docs, test_docs = set(), set(), set()
            for e in events:
                if e != event: #and e != 'ch':
                    train_users.update(fold_users[e])
                    train_docs.update(fold_docs[e])
                else: #elif event != 'ch':
                    test_users.update(fold_users[e])
                    test_docs.update(fold_docs[e])
               
            
            train_docs, train_users = list(train_docs), list(train_users)
            val_docs, val_users = list(val_docs), list(val_users)            
            test_docs, test_users = list(test_docs), list(test_users)
            print("\nTotal train users = ", len(train_users))
            print("Total train docs = ", len(train_docs))
            print("Total val users = ", len(val_users))
            print("Total val docs = ", len(val_docs))
            print("Total test users = ", len(test_users))
            print("Total test docs = ", len(test_docs))
            
            temp_dict = {}
            temp_dict['train_docs'] = train_docs
            temp_dict['val_docs'] = val_docs
            temp_dict['test_docs'] = test_docs
            doc_splits_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'doc_splits.json')
            print("Saving doc_splits dict in :", doc_splits_file)
            with open(doc_splits_file, 'w+') as j:
                json.dump(temp_dict, j)
            
            
            print("\nPreparing doc2id and user2id...")
            doc2id_train = {}
            node_type = []
            for train_count, doc in enumerate(train_docs):
                doc2id_train[str(doc)] = train_count
                node_type.append(1)
            print("Node_type = ", len(node_type))
            print("doc2id_train = ", len(doc2id_train))
                            
            
            a = set(train_users + val_users)
            b = set(test_users)
            print("\nUsers common between train/val and test = ", len(a.intersection(b)))

            all_users = list(set(train_users + val_users + test_users))                

            
            user2id_train = {}
            for count, user in enumerate(all_users):
                user2id_train[str(user)] = count + len(doc2id_train)
                node_type.append(2)
            user2id_train_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'user2id_train.json')
            print("user2id = ", len(user2id_train))
            print("Saving user2id_train in : ", user2id_train_file)
            with open(user2id_train_file, 'w+') as j:
                json.dump(user2id_train, j)
            
            
            node2id = doc2id_train.copy()
            node2id.update(user2id_train)
            print("node2id size = ", len(node2id))
            print("Node_type = ", len(node_type))
            node2id_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'node2id_lr_train_just.json')
            print("Saving node2id_lr_train in : ", node2id_file)
            with open(node2id_file, 'w+') as json_file:
                json.dump(node2id, json_file)
                       
            node_type_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'node_type_lr_train_just.npy')
            node_type = np.array(node_type)
            print(node_type.shape)
            print("Saving node_type in :", node_type_file)
            np.save(node_type_file, node_type, allow_pickle=True)
            
            
            print("\nAdding test docs..")
            orig_doc2id_len = len(doc2id_train)
            for test_count, doc in enumerate(test_docs):
                doc2id_train[str(doc)] = test_count + len(user2id_train) + orig_doc2id_len
            print("doc2id_train = ", len(doc2id_train))
            doc2id_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'doc2id_train_just.json')
            print("Saving doc2id dict in :", doc2id_file)
            with open(doc2id_file, 'w+') as j:
                json.dump(doc2id_train, j)
            
            node2id = doc2id_train.copy()
            node2id.update(user2id_train)
            print("node2id size = ", len(node2id))
            node2id_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'node2id_lr_just.json')
            print("Saving node2id_lr_train in : ", node2id_file)
            with open(node2id_file, 'w+') as json_file:
                json.dump(node2id, json_file)

            print("Done ! All files written..")
                    
        return None
                    
    


    def create_filtered_follower_following(self):
        for dataset in self.datasets:
            with open('./data/complete_data/user2id_pheme.json','r') as j:
               all_users = json.load(j)
            
            print("\n\n" + "-"*100 + "\n \t\t   Creating filtered follower-following\n" + '-'*100)
            user_contexts = ['user_followers', 'user_following']
            print_iter = int(len(all_users)/10)
            
            for user_context in user_contexts:
                print("    - from {}  folder...".format(user_context))
                src_dir2 = os.path.join(self.data_dir, 'complete_data', 'pheme', user_context)
                dest_dir = os.path.join(self.data_dir, 'complete_data', 'pheme', user_context+'_filtered')
                for root, dirs, files in os.walk(src_dir2):
                    for count, file in enumerate(files):
                        src_file_path = os.path.join(root, file)
                        # user_id = src_file_path.split(".")[0]
                        src_file = json.load(open(src_file_path, 'r'))
                        user_id = int(src_file['user_id'])
                        dest_file_path = os.path.join(dest_dir, str(user_id)+'.json')
                        if str(user_id) in all_users:
                            temp= set()
                            followers = src_file['followers'] if user_context == 'user_followers' else src_file['following']   
                            followers = list(map(int, followers))
                            for follower in followers:
                                if str(follower) in all_users:
                                    temp.update([follower])
                            temp_dict = {}
                            temp_dict['user_id'] = user_id
                            name = 'followers' if user_context == 'user_followers' else 'following'
                            temp_dict[name] = list(temp)
                            with open(dest_file_path, 'w+') as v:
                                json.dump(temp_dict, v)
                        else:
                            print("not found")
                        if count%print_iter==0:
                            # print("{}/{} done..  Non-zeros =  {}".format(count+1, num_users, adj_matrix.getnnz()))
                            print("{} done..".format(count+1))
                                 
        return None               
                    
                    
    
              
    def create_adj_matrix_tweet(self):
        # Commenters are connected to the source-tweet
        
        events = ['ch', 'ebola', 'ferg', 'german', 'gurlitt', 'ottawa', 'putin', 'sydney', 'toronto']
        
        for fold, event in enumerate(events):
            print("\n\n" + "-"*100 + "\n \t\t\tAnalyzing  PHEME dataset and fold {} for adj_matrix\n".format(fold+1) + '-'*100)
            
            user2id_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'user2id_train_just.json')
            doc2id_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'doc2id_train_just.json')
            
            user2id = json.load(open(user2id_file, 'r'))
            doc2id = json.load(open(doc2id_file, 'r'))
                               
            doc_splits_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'doc_splits.json')
            doc_splits = json.load(open(doc_splits_file, 'r'))
            test_docs = doc_splits['test_docs']
            train_docs = doc_splits['train_docs']
            val_docs = doc_splits['val_docs']
                            
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
            
            print("\nPreparing entries for doc-user and within commenter pairs...")
            # with open('./data/complete_data/doc2tags_{}.json'.format(dataset),'r') as j:
            #    doc2tags = json.load(j)
            src_dir = os.path.join(self.data_dir, 'complete')
            fold_docs = train_docs + val_docs
            for root, dirs, files in os.walk(src_dir):
                for count, file in enumerate(files):
                    src_file_path = os.path.join(root, file)
                    doc_key = file.split(".")[0]
                    src_file = json.load(open(src_file_path, 'r'))
                    users = src_file['users']
                    adj_matrix[doc2id[str(doc_key)], user2id[str(src_file['source_user'])]] = 1
                    adj_matrix[user2id[str(src_file['source_user'])], doc2id[str(doc_key)]] = 1
                    edge_type[doc2id[str(doc_key)], user2id[str(src_file['source_user'])]] = 2
                    edge_type[user2id[str(src_file['source_user'])], doc2id[str(doc_key)]] = 2
                    
                    # if doc_key in fold_docs:
                    for user in users:   
                        adj_matrix[doc2id[str(doc_key)], user2id[str(user)]] = 1
                        adj_matrix[user2id[str(user)], doc2id[str(doc_key)]] = 1
                        edge_type[doc2id[str(doc_key)], user2id[str(user)]] = 3
                        edge_type[user2id[str(user)], doc2id[str(doc_key)]] = 3
            

            # print("\nPreparing entries for doc-user and within commenter pairs...")
            # # with open('./data/complete_data/doc2tags_{}.json'.format(dataset),'r') as j:
            # #    doc2tags = json.load(j)
            # src_dir = os.path.join(self.data_dir, 'complete')
            # fold_docs = train_docs + val_docs
            # for root, dirs, files in os.walk(src_dir):
            #     for count, file in enumerate(files):
            #         src_file_path = os.path.join(root, file)
            #         doc_key = file.split(".")[0]
            #         src_file = json.load(open(src_file_path, 'r'))
            #         users = src_file['users']
            #         users.append(src_file['source_user'])
            #         # if doc_key in fold_docs:
            #         for user in users:   
            #             adj_matrix[doc2id[str(doc_key)], user2id[str(user)]] = 1
            #             adj_matrix[user2id[str(user)], doc2id[str(doc_key)]] = 1
                    
                    # # Add edges between commenters of a tweet
                    # for i in range(len(users)):
                    #     for j in range(i+1, len(users)):
                    #         adj_matrix[user2id[str(users[i])], user2id[str(users[j])]] = 1
                    #         adj_matrix[user2id[str(users[j])], user2id[str(users[i])]] = 1
                            
                    
                    # # Add edges between docs that use same hashtags as the current one
                    # common_tag_docs = []
                    # user_contexts = ['user_followers_filtered', 'users_following_filtered']
                    # for doc in doc2tags.keys():
                    #     if doc==doc_key:
                    #         continue
                    #     source_doc_tags = set(doc2tags[str(doc_key)])
                    #     target_doc_tags = set(doc2tags[str(doc)])
                    #     if len(source_doc_tags.intersection(target_doc_tags)) >0:
                    #         common_tag_docs.append(doc)
                            
                    # for common_doc in common_tag_docs:
                    #     adj_matrix[doc2id[str(doc_key)], doc2id[str(common_doc)]] = 1
                    #     adj_matrix[doc2id[str(common_doc)], doc2id[str(doc_key)]] = 1
                        
                    #     common_doc_file = os.path.join(root, str(common_doc)+'.json')
                    #     common_doc_file = json.load(open(common_doc_file, 'r'))
                    #     common_tag_users = common_doc_file['users']
                    #     for user in users: 
                    #         for common_tag_user in common_tag_users:
                    #             adj_matrix[user2id[str(common_tag_user)], user2id[str(user)]] = 1
                    #             adj_matrix[user2id[str(user)], user2id[str(common_tag_user)]] = 1
                        
                        
                        # for context in user_contexts:
                        #     common_doc_users_file = os.path.join(root, str(common_doc)+'.json')
                        #     common_doc_users = json.load(open(common_doc_users_file, 'r'))
                        #     users = common_doc_users['users']
                        #     for user in users:
                        #         additional_users_file = './data/complete_data/pheme/'+context+str(user)+'.json'
                        #         add_users = json.load(open(additional_users_file, 'r'))
                        #         followers = add_users['followers'] if context == 'user_followers_filtered' else add_users['following']   
                        #         followers = list(map(int, followers))
                        #         for follower in followers:
                        #             if follower in all_users:
                        #                 adj_matrix[doc2id[str(doc_key)], user2id[str(follower)]]=1
                        #                 adj_matrix[user2id[str(follower)], doc2id[str(doc_key)]]=1
                            
                if count%print_iter==0:
                    print("{} / {} done..".format(count+1, num_docs))


            end = time.time() 
            hrs, mins, secs = self.calc_elapsed_time(start, end)
            print("Done. Took {}hrs and {}mins and {}secs\n".format(hrs, mins, secs))  
            print("Non-zero entries = ", adj_matrix.count_nonzero())
            print("Non-zero entries edge_type = ", edge_type.count_nonzero())
            # print("Non-zero entries = ", len(np.nonzero(adj_matrix)[0]))
            
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
            
            filename = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'adj_matrix_just_lr.npz')
            # filename = self.data_dir+ '/complete_data' + '/adj_matrix_{}.npy'.format(dataset)
            print("\nMatrix construction done! Saving in  {}".format(filename))
            save_npz(filename, adj_matrix.tocsr())
            # np.save(filename, adj_matrix)
            
            filename = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'edge_type_just_lr.npz')
            print("\nEdge type saving in  {}".format(filename))
            save_npz(filename, edge_type.tocsr())
            
            # Creating an edge_list matrix of adj_matrix as required by some GCN frameworks
            print("\nCreating edge_index format of adj_matrix...")
            start = time.time()
            rows, cols = adj_matrix.nonzero()
            
            edge_index = np.vstack((np.array(rows), np.array(cols)))
            print("Edge index shape = ", edge_index.shape)
            
            edge_matrix_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'adj_matrix_just_lr_edge.npy')
            print("saving edge_list format in :  ", edge_matrix_file)
            np.save(edge_matrix_file, edge_index, allow_pickle=True)
            
            # rows, cols = edge_type.nonzero()
            # edge_index = np.vstack((np.array(rows), np.array(cols)))
            edge_index = edge_type[edge_type.nonzero()]
            edge_index = edge_index.toarray()
            edge_index = edge_index.squeeze(0)
            print("edge_type shape = ", edge_index.shape)
            edge_matrix_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'edge_type_just_lr_edge.npy')
            print("saving edge_type edge list format in :  ", edge_matrix_file)
            np.save(edge_matrix_file, edge_index, allow_pickle=True)
            
            
            hrs, mins, secs = self.calc_elapsed_time(start, time.time())
            print("Done. Took {}hrs and {}mins and {}secs\n".format(hrs, mins, secs))
        return None
    
    
    
    
    def create_adj_matrix_user(self):
        # Commenters are connected to the source-tweet
        
        events = ['ch', 'ebola', 'ferg', 'german', 'gurlitt', 'ottawa', 'putin', 'sydney', 'toronto']
        
        for fold, event in enumerate(events):
            print("\n\n" + "-"*100 + "\n \t\t\tAnalyzing  PHEME dataset and fold {} for adj_matrix\n".format(fold+1) + '-'*100)
            
            user2id_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'user2id_train.json')
            doc2id_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'doc2id_train.json')
            
            user2id = json.load(open(user2id_file, 'r'))
            doc2id = json.load(open(doc2id_file, 'r'))
                               
            doc_splits_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'doc_splits.json')
            doc_splits = json.load(open(doc_splits_file, 'r'))
            test_docs = doc_splits['test_docs']
            train_docs = doc_splits['train_docs']
            val_docs = doc_splits['val_docs']
                            
            num_users, num_docs = len(user2id), len(train_docs)+len(test_docs)
            print("\nNo.of unique users = ", num_users)
            print("No.of docs = ", num_docs)
            
            # Creating the adjacency matrix (doc-user edges)
            adj_matrix = lil_matrix((num_docs+num_users, num_users+num_docs))
            # adj_matrix = np.zeros((num_docs+num_users, num_users+num_docs))
            # adj_matrix_file = './data/complete_data/adj_matrix_pheme.npz'
            # adj_matrix = load_npz(adj_matrix_file)
            # adj_matrix = lil_matrix(adj_matrix)
            # Creating self-loops for each node (diagonals are 1's)
            for i in range(adj_matrix.shape[0]):
                adj_matrix[i,i] = 1
            print_iter = int(num_docs/10)
            print("\nSize of adjacency matrix = {} \nPrinting every  {} docs".format(adj_matrix.shape, print_iter))
            start = time.time()
            

            print("\nPreparing entries for doc-user and within commenter pairs...")
            # with open('./data/complete_data/doc2tags_{}.json'.format(dataset),'r') as j:
            #    doc2tags = json.load(j)
            src_dir = os.path.join(self.data_dir, 'complete')
            fold_docs = train_docs + val_docs
            for root, dirs, files in os.walk(src_dir):
                for count, file in enumerate(files):
                    src_file_path = os.path.join(root, file)
                    doc_key = file.split(".")[0]
                    src_file = json.load(open(src_file_path, 'r'))
                    users = src_file['users']
                    source_user = src_file['source_user']
                    # if doc_key in fold_docs:
                    adj_matrix[doc2id[str(doc_key)], user2id[str(source_user)]] = 1
                    adj_matrix[user2id[str(source_user)], doc2id[str(doc_key)]] = 1
                    for user in users:   
                        adj_matrix[user2id[str(source_user)], user2id[str(user)]] = 1
                        adj_matrix[user2id[str(user)], user2id[str(source_user)]] = 1
                    
                    # # Add edges between commenters of a tweet
                    # for i in range(len(users)):
                    #     for j in range(i+1, len(users)):
                    #         adj_matrix[user2id[str(users[i])], user2id[str(users[j])]] = 1
                    #         adj_matrix[user2id[str(users[j])], user2id[str(users[i])]] = 1
                            
                    
                    # # Add edges between docs that use same hashtags as the current one
                    # common_tag_docs = []
                    # user_contexts = ['user_followers_filtered', 'users_following_filtered']
                    # for doc in doc2tags.keys():
                    #     if doc==doc_key:
                    #         continue
                    #     source_doc_tags = set(doc2tags[str(doc_key)])
                    #     target_doc_tags = set(doc2tags[str(doc)])
                    #     if len(source_doc_tags.intersection(target_doc_tags)) >0:
                    #         common_tag_docs.append(doc)
                            
                    # for common_doc in common_tag_docs:
                    #     adj_matrix[doc2id[str(doc_key)], doc2id[str(common_doc)]] = 1
                    #     adj_matrix[doc2id[str(common_doc)], doc2id[str(doc_key)]] = 1
                        
                    #     common_doc_file = os.path.join(root, str(common_doc)+'.json')
                    #     common_doc_file = json.load(open(common_doc_file, 'r'))
                    #     common_tag_users = common_doc_file['users']
                    #     for user in users: 
                    #         for common_tag_user in common_tag_users:
                    #             adj_matrix[user2id[str(common_tag_user)], user2id[str(user)]] = 1
                    #             adj_matrix[user2id[str(user)], user2id[str(common_tag_user)]] = 1
                        
                        
                        # for context in user_contexts:
                        #     common_doc_users_file = os.path.join(root, str(common_doc)+'.json')
                        #     common_doc_users = json.load(open(common_doc_users_file, 'r'))
                        #     users = common_doc_users['users']
                        #     for user in users:
                        #         additional_users_file = './data/complete_data/pheme/'+context+str(user)+'.json'
                        #         add_users = json.load(open(additional_users_file, 'r'))
                        #         followers = add_users['followers'] if context == 'user_followers_filtered' else add_users['following']   
                        #         followers = list(map(int, followers))
                        #         for follower in followers:
                        #             if follower in all_users:
                        #                 adj_matrix[doc2id[str(doc_key)], user2id[str(follower)]]=1
                        #                 adj_matrix[user2id[str(follower)], doc2id[str(doc_key)]]=1
                            
                if count%print_iter==0:
                    print("{} / {} done..".format(count+1, num_docs))


            end = time.time() 
            hrs, mins, secs = self.calc_elapsed_time(start, end)
            print("Done. Took {}hrs and {}mins and {}secs\n".format(hrs, mins, secs))  
            print("Non-zero entries = ", adj_matrix.getnnz())
            # print("Non-zero entries = ", len(np.nonzero(adj_matrix)[0]))
            
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
            
            filename = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'adj_matrix_user_comments_lr.npz')
            # filename = self.data_dir+ '/complete_data' + '/adj_matrix_{}.npy'.format(dataset)
            print("\nMatrix construction done! Saving in  {}".format(filename))
            save_npz(filename, adj_matrix.tocsr())
            # np.save(filename, adj_matrix)
            
            # Creating an edge_list matrix of adj_matrix as required by some GCN frameworks
            print("\nCreating edge_index format of adj_matrix...")
            start = time.time()
            rows, cols = adj_matrix.nonzero()
            
            edge_index = np.vstack((np.array(rows), np.array(cols)))
            print("Edge index shape = ", edge_index.shape)
            
            edge_matrix_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'adj_matrix_user_comments_lr_edge.npy')
            print("saving edge_list format in :  ", edge_matrix_file)
            np.save(edge_matrix_file, edge_index, allow_pickle=True)
            
            
            hrs, mins, secs = self.calc_elapsed_time(start, time.time())
            print("Done. Took {}hrs and {}mins and {}secs\n".format(hrs, mins, secs))
        return None
    
    
    
    
    
    def create_feat_matrix(self, binary=True):
        events = ['ch', 'ebola', 'ferg', 'german', 'gurlitt', 'ottawa', 'putin', 'sydney', 'toronto']
        
        for fold, event in enumerate(events):
            print("\n\n" + "-"*100 + "\n \t\t\tAnalyzing  PHEME dataset and fold {} for feat_matrix\n".format(fold+1) + '-'*100)
            
            user2id_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'user2id_train_just.json')
            doc2id_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'doc2id_train_just.json')
            
            user2id = json.load(open(user2id_file, 'r'))
            doc2id = json.load(open(doc2id_file, 'r'))
                               
            doc_splits_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'doc_splits.json')
            doc_splits = json.load(open(doc_splits_file, 'r'))
            test_docs = doc_splits['test_docs']
            train_docs = doc_splits['train_docs']
            val_docs = doc_splits['val_docs']
                            
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
            src_doc_dir = os.path.join('./data', 'base_data', 'pheme_cv')
            for root, dirs, files in os.walk(src_doc_dir):
                for file in files:
                    if file.startswith('.') or file.startswith('structure') or file.startswith('annotation') or root.endswith('reactions'):
                        continue
                    else:
                        doc = file.split('.')[0]
                        if str(doc) not in test_docs:
                            src_file_path = os.path.join(root, file)
                            src_tweet = json.load(open(src_file_path, 'r'))
                            text = src_tweet['text'].lower()
                            text = re.sub(r'#[\w-]+', 'hashtag', text)
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
            vocab_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'vocab.json')
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
            src_doc_dir = os.path.join('./data', 'base_data', 'pheme_cv')
            for root, dirs, files in os.walk(src_doc_dir):
                for file in files:
                    if file.startswith('.') or file.startswith('structure') or file.startswith('annotation') or root.endswith('reactions'):
                        continue
                    else:
                        doc = root.split('\\')[-2]
                        # feat_matrix[doc2id[str(doc)], :] = np.random.random(len(vocab)) > 0.99
                        
                        src_file_path = os.path.join(root, file)
                        src_tweet = json.load(open(src_file_path, 'r'))
                        text = src_tweet['text'].lower()
                        text = re.sub(r'#[\w-]+', 'hashtag', text)
                        text = re.sub(r'https?://\S+', 'url', text)
                        text = text.replace('\n', ' ')
                        text = text.replace('\t', ' ')
                        # text = re.sub(r"[^A-Za-z(),!?\'`]", " ", text)
                        text = nltk.word_tokenize(text)
                        for token in text:
                            if token in vocab.keys():
                                feat_matrix[doc2id[str(doc)], vocab[token]] = 1
                        c+=1
                        if c%500 == 0:
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
            src_dir = os.path.join(self.data_dir,'complete')
            user_contexts = ['user_followers_filtered', 'user_following_filtered']
            for root, dirs, files in os.walk(src_dir):
                for count, file in enumerate(files):
                    print_iter = int(len(files) / 5)
                    src_file_path = os.path.join(root, file)
                    src_file = json.load(open (src_file_path, 'r'))
                    users = src_file['users']
                    users.append(src_file['source_user'])
                    doc_key = file.split(".")[0]
                    # if str(doc_key) in train_docs:
                    # Each user of this doc has its features as the features of the doc
                    if str(doc_key) in doc2id:
                        for user in users:
                            if str(user) in user2id:
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
            
            filename = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'feat_matrix_justnoise_lr.npz')
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
    
    
    
      
    def create_labels(self):
        """
        Create labels for each node of the graph
        """
        events = ['ch', 'ebola', 'ferg', 'german', 'gurlitt', 'ottawa', 'putin', 'sydney', 'toronto']
        
        for fold, event in enumerate(events):
            print("\n\n" + "-"*100 + "\n \t\t   Creating labels for fold {}\n".format(fold+1) + '-'*100)
            
            user2id_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'user2id_train_just.json')
            doc2id_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'doc2id_train_just.json')
            
            user2id = json.load(open(user2id_file, 'r'))
            doc2id = json.load(open(doc2id_file, 'r'))
                               
            doc_splits_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'doc_splits.json')
            doc_splits = json.load(open(doc_splits_file, 'r'))
            adj_matrix_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'adj_matrix_just_lr.npz')
            adj_matrix = load_npz(adj_matrix_file)
            N,_ = adj_matrix.shape
            del adj_matrix

            train_docs= doc_splits['train_docs']
            test_docs = doc_splits['test_docs']
            
            
            split_docs = train_docs + test_docs
                
            print("\nCreating doc2labels dictionary...")
            doc2labels = {}
            c=0
            pheme_dir = './data/base_data/pheme_cv'
            for root, dirs, files in os.walk(pheme_dir):
                for file in files:
                    if not file.startswith('annotation'):
                        continue
                    else:
                        src_file_path = os.path.join(root, file)
                        doc = root.split('\\')[-1]
                        with open(src_file_path, 'r') as j:
                            annotation = json.load(j)
                            doc2labels[str(doc)] = self.convert_annotations(annotation, string = False)
                            c+=1
                            if c%500 == 0:
                                print("{} done..".format(c))
            
            
            
            # print(len(doc2labels.keys()))
            # print(len(doc2id.keys()) - len(doc_splits['test_docs']))
            # assert len(doc2labels.keys()) == len(doc2id.keys()) - len(doc_splits['test_docs'])
            print("Len of doc2labels = {}\n".format(len(doc2labels)))
            labels_dict_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'doc2labels.json')
            print("Saving labels_dict for fold {} at:  {}".format(fold+1, labels_dict_file))
            with open(labels_dict_file, 'w+') as v:
                json.dump(doc2labels, v)
            
            labels_list = np.zeros(N)
            for key,value in doc2labels.items():
                labels_list[doc2id[str(key)]] = value
                          
            # Sanity Checks
            # print(sum(labels_list))
            # print(len(labels_list))
            # print(sum(labels_list[2402:]))
            # print(sum(labels_list[:2402]))
            
            filename = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'labels_list.json')
            temp_dict = {}
            temp_dict['labels_list'] = list(labels_list)
            print("Labels list construction done! Saving in :   {}".format(filename))
            with open(filename, 'w+') as v:
                json.dump(temp_dict, v)
            
            
            # Create the all_labels file
            all_labels = np.zeros(N)
            all_labels_data_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'all_labels.json')
            for doc in doc2labels.keys():
                all_labels[doc2id[str(doc)]] = doc2labels[str(doc)]
            
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
            
            user2id_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'user2id_train.json')
            doc2id_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'doc2id_train.json')
            split_mask_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'split_mask.json')
            adj_matrix_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'adj_matrix_lr.npz')
            doc_splits_file = os.path.join(self.data_dir, 'fold_{}'.format(fold+1), 'doc_splits.json')
            
            user2id = json.load(open(user2id_file, 'r'))
            doc2id = json.load(open(doc2id_file, 'r'))
            adj = load_npz(adj_matrix_file)
            N, _ = adj.shape
            del adj
                               
            doc_splits = json.load(open(doc_splits_file, 'r'))
            train_docs= doc_splits['train_docs']
            test_docs = doc_splits['test_docs']

            train_mask, test_mask = np.zeros(N), np.zeros(N) # np.zeros(test_n)
            repr_mask = np.ones(N)

            not_in_either=0
            train_labels, test_labels = [], []
            for doc, id in doc2id.items():
                if doc in train_docs:
                    train_mask[id] = 1
                elif doc in test_docs:
                    test_mask[id] = 1
                    repr_mask[id] = 0
                else:
                    not_in_either+=1
            
            print("\nnot_in_either = ", not_in_either)
            print("train_mask sum = ", sum(train_mask))
            print("test_mask sum = ", sum(test_mask))
            print("repr_mask sum = ", sum(repr_mask))
            
            
            temp_dict = {}
            temp_dict['test_mask'] = list(test_mask)
            temp_dict['train_mask'] = list(train_mask)
            temp_dict['repr_mask'] = list(repr_mask)
            with open(split_mask_file, 'w+') as j:
                json.dump(temp_dict, j)
            
                  
        return None
    
    
    
    def get_label_distribution_pheme(self, labels):  
        true = labels.count(1)
        false = labels.count(0)
        unverified = labels.count(2)
        denom = true + false + unverified    
        return true/denom, false/denom, unverified/denom

    def calc_elapsed_time(self, start, end):
        hours, rem = divmod(end-start, 3600)
        time_hours, time_rem = divmod(end, 3600)
        minutes, seconds = divmod(rem, 60)
        time_mins, _ = divmod(time_rem, 60)
        return int(hours), int(minutes), int(seconds)
                
        
                
        
      


if __name__== '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, default = './data/complete_data/pheme_cv',
                          help='path to dataset folder that contains the folders to gossipcop or politifact folders (raw data)') 
    
    parser.add_argument('--create_aggregate_folder', type = bool, default = False,
                          help='Aggregate only user ids from different folders of tweets/retweets to a single place')
    parser.add_argument('--create_doc_user_folds', type = bool, default = False,
                          help='create dictionaries of user_ids and doc_ids present in different CV folds')
    parser.add_argument('--create_dicts', type = bool, default = False,
                          help='Create doc2id and node2id dictionaries')
    parser.add_argument('--create_adj_matrix_tweet', type = bool, default = False,
                          help='create adjacency matrix for a given dataset with commenting users connected to source tweet')
    parser.add_argument('--create_adj_matrix_user', type = bool, default = False,
                          help='create adjacency matrix for a given dataset with commenting users connected to source user')
    parser.add_argument('--create_feat_matrix', type = bool, default = False,
                          help='To create feature matrix for a given dataset')
    parser.add_argument('--create_labels', type = bool, default = False,
                          help='To create labels for all the nodes')
    parser.add_argument('--create_split_masks', type = bool, default = True,
                          help='To create node masks for data splits')
    
    
    args, unparsed = parser.parse_known_args()
    config = args.__dict__
    
    preprocesser = GCN_PreProcess(config, pheme=True)