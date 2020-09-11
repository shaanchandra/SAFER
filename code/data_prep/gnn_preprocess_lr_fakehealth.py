import json, time, random, datetime
import os, sys, glob, csv, re, argparse
import numpy as np
import networkx as nx
from collections import defaultdict
from scipy.sparse import csr_matrix, lil_matrix, save_npz, load_npz
from collections import defaultdict

import nltk
# nltk.download('punkt')
sys.path.append("..")
import torch
from nltk.corpus import stopwords



class GCN_PreProcess():
    def __init__(self, config, release=False, story=False):
        self.config = config
        self.comp_dir = os.path.join('..', 'data', 'complete_data')
        self.data_dir = config['data_dir']
        self.datasets = []
        
        if release:
            self.datasets.append('HealthRelease')
        if story:
            self.datasets.append('HealthStory')
        
        if config['create_aggregate_folder']:
            self.create_aggregate_folder()
        if config['create_doc_user_splits']:
            self.create_doc_user_splits()
        if config['create_dicts']:
            self.create_dicts()
        if config['create_filtered_follower_following']:
            self.create_filtered_follower_following()
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
        """Create the 'complete' folder that has files as doc_IDs and contains alll the users that interacted with it """
        
        for dataset in self.datasets:
            print("\n" + "-"*100 + "\n \t\t Processing {} dataset for aggregating engagement info\n".format(dataset) + '-'*100)
            src_dir = os.path.join(self.data_dir, 'engagements', dataset)
            dest_dir = os.path.join(self.data_dir, 'engagements', 'complete', dataset)
            if not os.path.exists(dest_dir):
                print("Creating dir:  {}\n".format(dest_dir))
                os.makedirs(dest_dir)
            c=0   
            docs_done = defaultdict(list)
            for root, dirs, files in os.walk(src_dir):
                if root.endswith('replies'):
                    continue
                for count, file in enumerate(files):
                    if file.startswith('.'):
                        continue
                    c+=1
                    doc = root.split('\\')[-2]
                    src_file_path = os.path.join(root, file)
                    src_file = json.load(open(src_file_path, 'r'))
                    user = src_file['user']['id']
                    docs_done[doc].append(user)
                    if c%10000 == 0:
                        print("{} done".format(c))
                    
                                   
            print("\nTotal tweets/re-tweets in the data set = ", c)
            print("\nWriting all the info in the dir: ", dest_dir)
            for doc, user_list in docs_done.items():
                write_file = os.path.join(dest_dir, str(doc)+'.json')
                with open(write_file, 'w+') as j:
                    temp_dict = {}
                    temp_dict['users'] = list(set(user_list))
                    json.dump(temp_dict, j)
            print("\nDONE..!!")
                
        return None
    
    
    
    
    def create_doc_user_splits(self):
        """Create and save docs and users present in the data splits"""
        
        for dataset in self.datasets:
            print("\n\n" + "-"*100 + "\n \t\t   Creating doc and user splits for {}\n".format(dataset) + '-'*100)
            
            docsplits_file = os.path.join(self.data_dir, 'doc_splits_{}.json'.format(dataset)) 
            
            docsplits = json.load(open(docsplits_file, 'r'))
            train_docs = docsplits['train_docs']
            test_docs = docsplits['test_docs']
            val_docs = docsplits['val_docs']

            print("\nCreating users in splits file..")
            src_dir = os.path.join(self.data_dir, 'engagements', 'complete', dataset)
            train_users, val_users, test_users = set(), set(), set()
            for root, dirs, files in os.walk(src_dir):
                for count, file in enumerate(files):
                    doc_key = file.split('.')[0]
                    src_file_path = os.path.join(root, file)
                    src_file = json.load(open(src_file_path, 'r'))
                    users = src_file['users']
                    users = [int(s) for s in users if isinstance(s, int)]
                    if str(doc_key) in train_docs:
                        train_users.update(users)
                    if str(doc_key) in val_docs:
                        val_users.update(users)
                    if str(doc_key) in test_docs:
                        test_users.update(users)
            
            temp_dict = {}
            temp_dict['train_users'] = list(train_users)
            temp_dict['val_users'] = list(val_users)
            temp_dict['test_users'] = list(test_users)
            usersplits_file = os.path.join(self.data_dir, 'user_splits_{}.json'.format(dataset))
            print("user_splits written in : ", usersplits_file)
            with open(usersplits_file, 'w+') as j:
                json.dump(temp_dict, j)
                
        return None
    
    
    
    
    def create_dicts(self):
        """Create and save doc2id and node2id dicts""" 
        
        for dataset in self.datasets:
            print("\n\n" + "-"*100 + "\n \t\t\t   Creating dicts for  {} dataset \n".format(dataset) + '-'*100)
            
            print("\nCreating doc2id and user2id dicts....\n")
            user_splits_file = os.path.join(self.data_dir, 'user_splits_{}.json'.format(dataset))
            doc_splits_file = os.path.join(self.data_dir, 'doc_splits_{}.json'.format(dataset))
            
            user_splits = json.load(open(user_splits_file, 'r'))
            doc_splits = json.load(open(doc_splits_file, 'r'))
            
            train_users = user_splits['train_users']
            val_users = user_splits['val_users']
            test_users = user_splits['test_users']
            
            train_docs = doc_splits['train_docs']
            val_docs = doc_splits['val_docs']
            test_docs = doc_splits['test_docs']       
            
            doc2id_train = {}
            node_type = []
            for train_count, doc in enumerate(train_docs):
                doc2id_train[str(doc)] = train_count
                node_type.append(1)
            print("Train docs = ", len(train_docs))
            print("Node_type = ", len(node_type))


            for val_count, doc in enumerate(val_docs):
                doc2id_train[str(doc)] = val_count + len(train_docs)
                node_type.append(1)
            print("Val docs = ", len(val_docs))
            print("doc2id_train = ", len(doc2id_train))
            print("Node_type = ", len(node_type))
            doc2id_file = os.path.join(self.comp_dir, dataset, 'doc2id_lr_top50_train.json')
            print("Saving doc2id dict in :", doc2id_file)
            with open(doc2id_file, 'w+') as j:
                json.dump(doc2id_train, j)
                            
            
            print('\nTrain users = ', len(train_users))
            print("Test users = ", len(test_users))
            print("Val users = ", len(val_users))
            # print("All users = ", len(all_users))
            
            # If do not want to include the most frequent users
            # if dataset == 'HealthStory':
            restricted_users_file = os.path.join(self.data_dir, 'restricted_users_5_{}.json'.format(dataset))
            valid_users_file = os.path.join(self.data_dir, 'valid_users_top50_{}.json'.format(dataset))
            restricted_users = json.load(open(restricted_users_file, 'r'))['restricted_users']
            valid_users = json.load(open(valid_users_file, 'r'))['valid_users']
            # train_users = [u for u in train_users if str(u) not in restricted_users]
            # val_users = [u for u in val_users if str(u) not in restricted_users]
            # test_users = [u for u in test_users if str(u) not in restricted_users]
            
            train_users = [u for u in train_users if str(u) not in restricted_users and str(u) in valid_users]
            val_users = [u for u in val_users if str(u) not in restricted_users and str(u) in valid_users]
            test_users = [u for u in test_users if str(u) not in restricted_users and str(u) in valid_users]
            
            # train_users = [u for u in train_users if str(u) in valid_users]
            # val_users = [u for u in val_users if str(u) in valid_users]
            # test_users = [u for u in test_users if str(u) in valid_users]
                
         
            # train_users = [u for u in train_users if u in done_users['done_users']]
            # val_users = [u for u in val_users if u in done_users['done_users']]
            # test_users = [u for u in test_users if u in done_users['done_users']]
            
            # train_users = list(set(train_users) - set(done_users['done_users']) - set(restricted_users['restricted_users']))
            
            a = set(train_users + val_users)
            b = set(test_users)
            print("\nUsers common between train/val and test = ", len(a.intersection(b)))

            all_users = list(set(train_users + val_users + test_users))                

            print('\nTrain users = ', len(train_users))
            print("Test users = ", len(test_users))
            print("Val users = ", len(val_users))
            print("All users = ", len(all_users))
            
            user2id_train = {}
            for count, user in enumerate(all_users):
                user2id_train[str(user)] = count + len(doc2id_train)
                node_type.append(2)
            user2id_train_file = os.path.join(self.comp_dir, dataset, 'user2id_lr_top50_train.json')
            print("user2_id = ", len(user2id_train))
            print("Saving user2id_train in : ", user2id_train_file)
            with open(user2id_train_file, 'w+') as j:
                json.dump(user2id_train, j)
            
            
            node2id = doc2id_train.copy()
            node2id.update(user2id_train)
            print("node2id size = ", len(node2id))
            print("Node_type = ", len(node_type))
            node2id_file = os.path.join(self.comp_dir, dataset, 'node2id_lr_top50_train.json')
            print("Saving node2id_lr_train in : ", node2id_file)
            with open(node2id_file, 'w+') as json_file:
                json.dump(node2id, json_file)
            
            node_type_file = os.path.join(self.comp_dir, dataset, 'node_type_lr_top50_train.npy')
            node_type = np.array(node_type)
            print(node_type.shape)
            print("Saving node_type in :", node_type_file)
            np.save(node_type_file, node_type, allow_pickle=True)
            
            print("\nAdding test docs..")
            orig_doc2id_len = len(doc2id_train)
            for test_count, doc in enumerate(test_docs):
                doc2id_train[str(doc)] = test_count + len(user2id_train) + orig_doc2id_len
            print("Test docs = ", len(test_docs))
            print("doc2id_train = ", len(doc2id_train))
            with open(os.path.join(self.comp_dir, dataset, 'doc2id_lr_top50_train.json'), 'w+') as j:
                json.dump(doc2id_train, j)
            
            node2id = doc2id_train.copy()
            node2id.update(user2id_train)
            print("node2id size = ", len(node2id))
            node2id_file = os.path.join(self.comp_dir, dataset, 'node2id_lr_top50.json')
            print("Saving node2id_lr in : ", node2id_file)
            with open(node2id_file, 'w+') as json_file:
                json.dump(node2id, json_file)

            print("Done ! All files written..")
                    
        return None
                    
                    
    

    def create_filtered_follower_following(self):
        """Creates a filtered lists of followers and following list of each user comprising of only those users that are in the dataset """
        
        for dataset in self.datasets:
            with open('./data/complete_data/{}/user2id_lr_top50_train.json'.format(dataset),'r') as j:
               all_users = json.load(j)
            
            print("\n\n" + "-"*100 + "\n \t\t   Creating filtered follower-following for {}\n".format(dataset) + '-'*100)
            user_contexts = ['user_followers', 'user_following']
            print_iter = int(len(all_users)/10)
            print("Total users in this dataset = ", len(all_users))
            
            for user_context in user_contexts:
                print("    - from {}  folder...".format(user_context))
                src_dir2 = os.path.join(self.data_dir, user_context)
                dest_dir = os.path.join(self.data_dir, user_context+'_filtered')
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)
                for root, dirs, files in os.walk(src_dir2):
                    for count, file in enumerate(files):
                        print_iter = int(len(files)/10)
                        if count == 0:
                            print("Total user files = ", len(files))
                            print("Writing filtered lists in : ", dest_dir)
                            print("Printing every: ", print_iter)
                        src_file_path = os.path.join(root, file)
                        user_id = file.split(".")[0]
                        if os.path.isfile(os.path.join(dest_dir, str(user_id)+'.json')):
                            continue
                        if str(user_id) in all_users:
                            src_file = json.load(open(src_file_path, 'r'))
                            dest_file_path = os.path.join(dest_dir, str(user_id)+'.json')
                            temp= set()
                            followers = src_file['ids']
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

                        if count%print_iter==0:
                            # print("{}/{} done..  Non-zeros =  {}".format(count+1, num_users, adj_matrix.getnnz()))
                            print("{} done..".format(count+1))
                                
        return None



                
                    
                   
    def create_adj_matrix(self):
        """create and save adjacency matrix of the community graph"""
        
        for dataset in self.datasets:
            print("\n\n" + "-"*100 + "\n \t\t\tProcessing  {} dataset for adj_matrix\n".format(dataset) + '-'*100)
            
            user2id_file = os.path.join(self.comp_dir, dataset, 'user2id_lr_top50_train.json')
            doc2id_file = os.path.join(self.comp_dir, dataset, 'doc2id_lr_top50_train.json')
            doc_splits_file = os.path.join(self.data_dir, 'doc_splits_{}.json'.format(dataset))
            user2id = json.load(open(user2id_file,'r'))
            doc2id = json.load(open(doc2id_file,'r'))
            doc_splits = json.load(open(doc_splits_file, 'r'))
            test_docs = doc_splits['test_docs']
                            
            num_users, num_docs = len(user2id), len(doc2id)-len(test_docs)
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
            src_dir = os.path.join(self.data_dir, 'engagements', 'complete', dataset)
            not_found=0
            for root, dirs, files in os.walk(src_dir):
                for count, file in enumerate(files):
                    src_file_path = os.path.join(root, file)
                    doc_key = file.split(".")[0]
                    src_file = json.load(open(src_file_path, 'r'))
                    users = src_file['users']
                    for user in users:  
                        if str(doc_key) in doc2id and str(user) in user2id and str(doc_key) not in test_docs:
                            adj_matrix[doc2id[str(doc_key)], user2id[str(user)]] = 1
                            adj_matrix[user2id[str(user)], doc2id[str(doc_key)]] = 1
                            edge_type[doc2id[str(doc_key)], user2id[str(user)]] = 2
                            edge_type[user2id[str(user)], doc2id[str(doc_key)]] = 2
                        else:
                            not_found+=1


            end = time.time() 
            hrs, mins, secs = self.calc_elapsed_time(start, end)
            print("Done. Took {}hrs and {}mins and {}secs\n".format(hrs, mins, secs))  
            print("Not Found users = ", not_found)
            print("Non-zero entries = ", adj_matrix.getnnz())
            print("Non-zero entries edge_type = ", edge_type.getnnz())
            # print("Non-zero entries = ", len(np.nonzero(adj_matrix)[0]))
            
            # Creating the adjacency matrix (user-user edges)
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
                                    adj_matrix[user2id[str(user_id)], user2id[str(follower)]]= 1
                                    adj_matrix[user2id[str(follower)], user2id[str(user_id)]]= 1
                                    edge_type[user2id[str(user_id)], user2id[str(follower)]]= 3
                                    edge_type[user2id[str(follower)], user2id[str(user_id)]]= 3
                                    
                        else:
                            not_found +=1
                        # if count%print_iter==0:
                        #     # print("{}/{} done..  Non-zeros =  {}".format(count+1, num_users, adj_matrix.getnnz()))
                        #     print("{}/{} done..  Non-zeros =  {}".format(count+1, num_users, len(np.nonzero(adj_matrix)[0])))
                                         
            hrs, mins, secs = self.calc_elapsed_time(start, time.time())
            print("Done. Took {}hrs and {}mins and {}secs\n".format(hrs, mins, secs))
            print("Not found user_ids = ", not_found)
            print("Total Non-zero entries = ", adj_matrix.getnnz())
            print("Total Non-zero entries edge_type = ", edge_type.getnnz())
            # print("Total Non-zero entries = ", len(np.nonzero(adj_matrix)[0]))
            
            filename = os.path.join(self.comp_dir, dataset, 'adj_matrix_lr_top50_train.npz')
            # filename = self.data_dir+ '/complete_data' + '/adj_matrix_{}.npy'.format(dataset)
            print("\nMatrix construction done! Saving in  {}".format(filename))
            save_npz(filename, adj_matrix.tocsr())
            # np.save(filename, adj_matrix)
            
            filename = os.path.join(self.comp_dir, dataset, 'edge_type_lr_top50.npz')
            print("\nEdge_type construction done! Saving in  {}".format(filename))
            save_npz(filename, edge_type.tocsr())
            
            # Creating an edge_list matrix of the adj_matrix as required by some GCN frameworks
            print("\nCreating edge_index format of adj_matrix...")
            start = time.time()
            rows, cols = adj_matrix.nonzero()
            
            edge_index = np.vstack((np.array(rows), np.array(cols)))
            print("Edge index shape = ", edge_index.shape)
            
            edge_matrix_file = os.path.join(self.comp_dir, dataset, 'adj_matrix_lr_top50_train_edge.npy')
            print("saving edge_list format in :  ", edge_matrix_file)
            np.save(edge_matrix_file, edge_index, allow_pickle=True)
            
            edge_index = edge_type[edge_type.nonzero()]
            edge_index = edge_index.toarray()
            edge_index = edge_index.squeeze(0)
            print("edge_type shape = ", edge_index.shape)
            edge_matrix_file = os.path.join(self.comp_dir, dataset, 'edge_type_lr_top50_edge.npy')
            print("saving edge_type edge list format in :  ", edge_matrix_file)
            np.save(edge_matrix_file, edge_index, allow_pickle=True)
            
            
            hrs, mins, secs = self.calc_elapsed_time(start, time.time())
            print("Done. Took {}hrs and {}mins and {}secs\n".format(hrs, mins, secs))
        return None
    
    
    
    def create_feat_matrix(self, binary=True):
        """ Create and save the initial node representations of each node of the graph """
        for dataset in self.datasets:
            print("\n\n" + "-"*100 + "\n \t\tProcessing  {} dataset  for feature_matrix\n".format(dataset) + '-'*100)
            user2id_file = os.path.join(self.comp_dir, dataset, 'user2id_lr_top50_train.json')
            doc2id_file = os.path.join(self.comp_dir, dataset, 'doc2id_lr_top50_train.json')
            doc_splits_file = os.path.join(self.data_dir, 'doc_splits_{}.json'.format(dataset))
            user_splits_file = os.path.join(self.data_dir, 'user_splits_{}.json'.format(dataset))
            
            user2id = json.load(open(user2id_file,'r'))
            doc2id = json.load(open(doc2id_file,'r'))
            doc_splits = json.load(open(doc_splits_file, 'r'))
            user_splits = json.load(open(user_splits_file, 'r'))
            
            train_users = user_splits['train_users']
            val_users = user_splits['val_users']
            
            train_docs= doc_splits['train_docs']
            val_docs = doc_splits['val_docs']
            test_docs = doc_splits['test_docs']
            
            N = len(train_docs) + len(val_docs) + len(user2id)
            
            vocab = {}
            vocab_size=0
            start = time.time()
            stop_words = set(stopwords.words('english'))
            print("\nBuilding vocabulary...") 
            src_doc_dir = os.path.join(self.data_dir, 'content', dataset+"/*.json")
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
                    text_filtered = [w for w in text if not w in stop_words]
                    for token in text_filtered:
                        if token not in vocab.keys():
                            vocab[token] = vocab_size
                            vocab_size+=1
                
    
            hrs, mins, secs = self.calc_elapsed_time(start, time.time())
            print("Done. Took {}hrs and {}mins and {}secs\n".format(hrs, mins, secs))
            print("Size of vocab =  ", vocab_size)
            vocab_file = os.path.join(self.comp_dir, dataset, 'vocab.json')
            print("Saving vocab for  {}  at:  {}".format(dataset, vocab_file))
            with open(vocab_file, 'w+') as v:
                json.dump(vocab, v)
            
            
            feat_matrix = lil_matrix((N, vocab_size))
            print("\nSize of feature matrix = ", feat_matrix.shape)
            print("\nCreating feat_matrix entries for docs nodes...")
            start = time.time()
            split_docs = train_docs+val_docs

            src_doc_dir = os.path.join(self.data_dir, 'content', dataset+"/*.json")
            all_files = glob.glob(src_doc_dir)
            for count, file in enumerate(all_files):
                print_iter = int(len(all_files) / 5)
                doc_name = file.split('\\')[-1].split('.')[0]
                if str(doc_name) in split_docs:
                    if str(doc_name) in doc2id and str(doc_name) not in test_docs:
                        # feat_matrix[doc2id[str(doc_name)], :] = np.random.random(len(vocab)) > 0.99
                        
                        with open(file, 'r') as f:
                            vector = np.zeros(len(vocab))
                            file_content = json.load(f)
                            text = file_content['text'].replace('\n', ' ')[:1500]
                            text = text.replace('\t', ' ')
                            text= text.lower()
                            text = re.sub(r'#[\w-]+', 'hashtag', text)
                            text = re.sub(r'https?://\S+', 'url', text)
                            # text = re.sub(r"[^A-Za-z(),!?\'`]", " ", text)
                            text = nltk.word_tokenize(text)
                            text_filtered = [w for w in text if not w in stop_words]
                            for token in text_filtered:
                                if token in vocab.keys():
                                    vector[vocab[token]] = 1
                            feat_matrix[doc2id[str(doc_name)], :] = vector
                if count%print_iter==0:
                    print("{} / {} done..".format(count+1, len(all_files)))
                
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
            src_dir = os.path.join(self.data_dir, 'engagements', 'complete', dataset)
            user_contexts = ['user_followers_filtered', 'user_following_filtered']
            for root, dirs, files in os.walk(src_dir):
                for count, file in enumerate(files):
                    print_iter = int(len(files) / 10)
                    src_file_path = os.path.join(root, file)
                    src_file = json.load(open (src_file_path, 'r'))
                    users = src_file['users']
                    doc_key = file.split(".")[0]
                    # if str(doc_key) in train_docs:
                    # Each user of this doc has its features as the features of the doc
                    if (str(doc_key) in split_docs) and str(doc_key) in doc2id:
                        for user in users:
                            if str(user) in user2id:
                                feat_matrix[user2id[str(user)], :] += feat_matrix[doc2id[str(doc_key)], :]
                                
                            
                    if count%print_iter==0:
                        print(" {} / {} done..".format(count+1, len(files)))
                        print(datetime.datetime.now())
                    
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
            
            filename = os.path.join(self.comp_dir, dataset, 'feat_matrix_lr_top50_train.npz')
            print("Matrix construction done! Saving in :   {}".format(filename))
            save_npz(filename, feat_matrix.tocsr())
        return None
            
            
    
      
    def create_labels(self):
        """
        Create labels for each node of the graph
        """
        for dataset in self.datasets:
            print("\n\n" + "-"*100 + "\n \t\t   Processing  {} dataset  for Creating Labels\n".format(dataset) + '-'*100)
            doc2id_file = os.path.join(self.comp_dir, dataset, 'doc2id_lr_top50_train.json')
            adj_matrix_file = os.path.join(self.comp_dir, dataset, 'adj_matrix_lr_top50_train.npz')
            
            doc2id = json.load(open(doc2id_file, 'r')) 
            adj_matrix = load_npz(adj_matrix_file)
            N,_ = adj_matrix.shape
            del adj_matrix
            
            doc_splits_file = os.path.join(self.data_dir, 'doc_splits_{}.json'.format(dataset)) 
            doc_splits = json.load(open(doc_splits_file, 'r'))
            train_docs= doc_splits['train_docs']
            val_docs = doc_splits['val_docs']
            
            split_docs = train_docs + val_docs
                
            print("\nCreating doc2labels dictionary...")
            doc2labels = {}
            src_dir = os.path.join(self.data_dir, 'reviews', dataset+'.json')
            doc_labels = json.load(open(src_dir, 'r'))
            for count, doc in enumerate(doc_labels):
                if str(doc['news_id']) in split_docs:
                    label = 1 if doc['rating'] < 3 else 0  # rating less than 3 is fake
                    doc2labels[str(doc['news_id'])] = label
            
            print(len(doc2labels.keys()))
            print(len(doc2id.keys()) - len(doc_splits['test_docs']))
            assert len(doc2labels.keys()) == len(doc2id.keys()) - len(doc_splits['test_docs'])
            print("Len of doc2labels  = {}\n".format(len(doc2labels)))
            labels_dict_file = os.path.join(self.comp_dir, dataset, 'doc2labels_lr_top50_train.json')
            print("Saving labels_dict for  {} at:  {}".format(dataset, labels_dict_file))
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
            
            filename = os.path.join(self.comp_dir, dataset, 'labels_list_lr_top50_train.json')
            temp_dict = {}
            temp_dict['labels_list'] = list(labels_list)
            print("Labels list construction done! Saving in :   {}".format(filename))
            with open(filename, 'w+') as v:
                json.dump(temp_dict, v)
            
            
            # Create the all_labels file
            all_labels = np.zeros(N)
            all_labels_data_file = os.path.join(self.comp_dir, dataset, 'all_labels_lr_top50_train.json') 
            for doc in doc2labels.keys():
                all_labels[doc2id[str(doc)]] = doc2labels[str(doc)]
            
            temp_dict = {}
            temp_dict['all_labels'] = list(all_labels)
            print("Sum of labels this test set = ", sum(all_labels))
            print("Len of labels = ", len(all_labels))
            print("Writing all labels in: ", all_labels_data_file)
            with open(all_labels_data_file, 'w+') as j:
                json.dump(temp_dict, j)
        return None
    
    
    def create_split_masks(self):
        """create and save node masks for the train and val article nodes"""
        
        for dataset in self.datasets:
            print("\n\n" + "-"*100 + "\n \t\t   Creating split masks for {}\n".format(dataset) + '-'*100)
            
            doc2id_file = os.path.join(self.comp_dir, dataset, 'doc2id_lr_top50_train.json') 
            doc_splits_file = os.path.join(self.data_dir, 'doc_splits_{}.json'.format(dataset))
            train_adj_matrix_file = os.path.join(self.comp_dir, dataset, 'adj_matrix_lr_top50_train.npz')
            
            doc2id = json.load(open(doc2id_file, 'r'))
            doc_splits = json.load(open(doc_splits_file, 'r'))
            train_adj = load_npz(train_adj_matrix_file)
            
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
            split_mask_file = os.path.join(self.comp_dir, dataset, 'split_mask_top50.json')
            print("Writing split mask file in : ", split_mask_file)
            with open(split_mask_file, 'w+') as j:
                json.dump(temp_dict, j)               
                  
        return None
    
    
   

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
                
        
                
        
      


if __name__== '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, default = '../FakeHealth',
                          help='path to dataset folder that contains the folders to gossipcop or politifact folders (raw data)')
    
    
    parser.add_argument('--create_aggregate_folder', type = bool, default = False,
                          help='Aggregate only user ids from different folders of tweets/retweets to a single place')
    parser.add_argument('--create_doc_user_splits', type = bool, default = False,
                          help='Create splits of users and docs')
    parser.add_argument('--create_dicts', type = bool, default = False,
                          help='Create doc2id and node2id dictionaries')
    parser.add_argument('--create_filtered_follower_following', type = bool, default = False,
                          help='Reduce the follow-er/ing list to users that are in the dataset')
    parser.add_argument('--create_adj_matrix', type = bool, default = False,
                          help='To create adjacency matrix for a given dataset')
    parser.add_argument('--create_feat_matrix', type = bool, default = True,
                          help='To create feature matrix for a given dataset')
    parser.add_argument('--create_labels', type = bool, default = False,
                          help='To create labels for all the nodes')
    parser.add_argument('--create_split_masks', type = bool, default = False,
                          help='To create node masks for data splits')
    
    
    args, unparsed = parser.parse_known_args()
    config = args.__dict__
    
    preprocesser = GCN_PreProcess(config, release=False, story = True)