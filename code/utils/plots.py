import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json, os, argparse
from collections import defaultdict
from math import sqrt

import torch
from scipy.sparse import csr_matrix, lil_matrix, save_npz, load_npz
from torch_geometric.data import Data
from torch_geometric.data.cluster import ClusterData, ClusterLoader
from torch_geometric.utils import remove_isolated_nodes, contains_isolated_nodes, contains_self_loops
from torch_geometric.utils import k_hop_subgraph, to_networkx
import networkx as nx

from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
sys.path.append("..")





def plot_tsne_docs_news(config):
            
    base_dir = config['base_dir']
    doc2labels_test_file = os.path.join(base_dir, 'cached_embeds', 'docs_labels_lr_30_30_test.json')
    gnn_doc_embeds_file = os.path.join(base_dir, 'cached_embeds', 'doc_embeds_graph_lr_30_30_test_3_{}.pt'.format(config['model_name']))
    text_doc_embeds_file = os.path.join(base_dir, 'doc_embeds_roberta_lr_test.pt')
    text2id_file = os.path.join(base_dir, 'doc2id_encoder.json')
    node2id_file = os.path.join(os.getcwd(), '..', 'data', 'complete_data', 'node2id_lr_30_30_gossipcop.json')
    node2id = json.load(open(node2id_file, 'r'))
    text2id = json.load(open(text2id_file, 'r'))
    doc2labels_test = json.load(open(doc2labels_test_file, 'r'))
    gnn_doc_embeds_loaded = torch.load(gnn_doc_embeds_file)
    text_doc_embeds_loaded = torch.load(text_doc_embeds_file, map_location=torch.device('cpu'))
    
    print(gnn_doc_embeds_loaded.shape, text_doc_embeds_loaded.shape)
    
    row_sum = gnn_doc_embeds_loaded.sum(1)
    row_sum = list(row_sum)
    c=0
    for s in row_sum:
        if s==0:
            c+=1
    print("Zero entries gnn embeds = ", c)
    row_sum = text_doc_embeds_loaded.sum(1)
    row_sum = list(row_sum)
    c=0
    for s in row_sum:
        if s==0:
            c+=1
    print("Zero entries text embeds = ", c)
    
    
    if config['mode'] == 'gnn':
        doc_embeds = torch.zeros(len(doc2labels_test), gnn_doc_embeds_loaded.shape[1])
        print(doc_embeds.shape)
        labels = []
        doc2id = {}
        c=0
        for node, id in node2id.items():
            if node in doc2labels_test:
                labels.append(doc2labels_test[str(node)])
                doc_embeds[c, :] = gnn_doc_embeds_loaded[id, :]
                doc2id[str(node)] = c
                c+=1
        
        
        
    if config['mode'] == 'text':
        doc_embeds = torch.zeros(len(doc2labels_test), text_doc_embeds_loaded.shape[1])
        print(doc_embeds.shape)
        labels = []
        doc2id = {}
        c=0
        for node, id in text2id.items():
            if node in doc2labels_test:
                labels.append(doc2labels_test[str(node)])
                doc_embeds[c, :] = text_doc_embeds_loaded[id, :]
                doc2id[str(node)] = c
                c+=1
                
    
    
    if config['mode'] == 'gnn+text':
        doc_embeds = torch.zeros(len(doc2labels_test), text_doc_embeds_loaded.shape[1] + gnn_doc_embeds_loaded.shape[1])
        print(doc_embeds.shape)
        labels = []
        doc2id = {}
        c=0
        for node, id in text2id.items():
            if node in doc2labels_test and node in node2id:
                labels.append(doc2labels_test[str(node)])
                gnn_embeds = gnn_doc_embeds_loaded[node2id[str(node)], :]
                text_embeds = text_doc_embeds_loaded[id, :]
                doc_embeds[c, :] = torch.cat((gnn_embeds.unsqueeze(1), text_embeds.unsqueeze(1))).squeeze(1)
                doc2id[str(node)] = c
                c+=1
                
                
        
    print(c)
    print("Real label users = ", labels.count(0))
    print("Fake label users = ", labels.count(1))
    print("doc2id = ", len(doc2id))
    row_sum = doc_embeds.sum(1)
    row_sum = list(row_sum)
    c=0
    for s in row_sum:
        if s==0:
            c+=1
    print("Zero entries in final doc_embeds = ", c)
    
    
    tsne_file = os.path.join(base_dir, 'tsne', 'doc_tsne.npy')
    print("\nRunning TSNE..")
    doc_embeds = doc_embeds.numpy()
    doc_tsne = TSNE(n_components = config['reduc_dim'], init= 'random', perplexity = 50, learning_rate=200, early_exaggeration= 12).fit_transform(doc_embeds)
    print(doc_tsne.shape)
    np.save(tsne_file, doc_tsne, allow_pickle=True)
    
    if config['reduc_dim']==2:
        fig, ax = plt.subplots()
    elif config['reduc_dim']==3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
    for label in [0, 1]:
        x , y = [], []
        z=[]
        indexer = node2id if config['mode'] == 'gnn' else text2id
        for node, id in indexer.items():
            if node in doc2labels_test:
                if doc2labels_test[str(node)] == label:
                    x.append(doc_tsne[doc2id[str(node)], 0])
                    y.append(doc_tsne[doc2id[str(node)], 1])
                    if config['reduc_dim']==3:
                        z.append(doc_tsne[doc2id[str(node)], 2])
        
        color = 'black' if label==0 else 'red'
        legend = 'Real' if label==0 else 'Fake'
        
        if config['reduc_dim'] ==3:
            ax.scatter(x, y, z, c=color, label=legend,  alpha=0.6)
        else:
            ax.scatter(x, y, c=color, label=legend,  alpha=0.6)
            
            
    ax.legend(loc=2, prop={'size': 20})
    plt.show()   



def plot_tsne_docs_health(config):
            
    base_dir = config['base_dir']
    doc2labels_test_file = os.path.join(base_dir, 'cached_embeds', 'docs_labels_lr_test.json')
    gnn_doc_embeds_file = os.path.join(base_dir, 'cached_embeds', 'doc_embeds_graph_top10_lr_test_3_{}.pt'.format(config['model_name']))
    text_doc_embeds_file = os.path.join(base_dir, 'cached_embeds', 'doc_embeds_roberta_3_test.pt')
    text2id_file = os.path.join(base_dir, 'doc2id_encoder.json')
    node2id_file = os.path.join(base_dir, 'node2id_lr_top10.json')
    node2id = json.load(open(node2id_file, 'r'))
    text2id = json.load(open(text2id_file, 'r'))
    doc2labels_test = json.load(open(doc2labels_test_file, 'r'))
    gnn_doc_embeds_loaded = torch.load(gnn_doc_embeds_file)
    text_doc_embeds_loaded = torch.load(text_doc_embeds_file, map_location=torch.device('cpu'))
    
    print(gnn_doc_embeds_loaded.shape, text_doc_embeds_loaded.shape)
    
    row_sum = gnn_doc_embeds_loaded.sum(1)
    row_sum = list(row_sum)
    c=0
    for s in row_sum:
        if s==0:
            c+=1
    print("Zero entries gnn embeds = ", c)
    row_sum = text_doc_embeds_loaded.sum(1)
    row_sum = list(row_sum)
    c=0
    for s in row_sum:
        if s==0:
            c+=1
    print("Zero entries text embeds = ", c)
    
    
    if config['mode'] == 'gnn':
        doc_embeds = torch.zeros(len(doc2labels_test), gnn_doc_embeds_loaded.shape[1])
        labels = []
        doc2id = {}
        c=0
        for node, id in node2id.items():
            if node in doc2labels_test:
                labels.append(doc2labels_test[str(node)])
                doc_embeds[c, :] = gnn_doc_embeds_loaded[id, :]
                doc2id[str(node)] = c
                c+=1
        
        
        
    if config['mode'] == 'text':
        doc_embeds = torch.zeros(len(doc2labels_test), text_doc_embeds_loaded.shape[1])
        print(doc_embeds.shape)
        labels = []
        doc2id = {}
        c=0
        for node, id in text2id.items():
            if node in doc2labels_test:
                labels.append(doc2labels_test[str(node)])
                doc_embeds[c, :] = text_doc_embeds_loaded[id, :]
                doc2id[str(node)] = c
                c+=1
                
    
    
    if config['mode'] == 'gnn+text':
        doc_embeds = torch.zeros(len(doc2labels_test), text_doc_embeds_loaded.shape[1] + gnn_doc_embeds_loaded.shape[1])
        print(doc_embeds.shape)
        labels = []
        doc2id = {}
        c=0
        for node, id in text2id.items():
            if node in doc2labels_test and node in node2id:
                labels.append(doc2labels_test[str(node)])
                gnn_embeds = gnn_doc_embeds_loaded[node2id[str(node)], :]
                text_embeds = text_doc_embeds_loaded[id, :]
                doc_embeds[c, :] = torch.cat((gnn_embeds.unsqueeze(1), text_embeds.unsqueeze(1))).squeeze(1)
                doc2id[str(node)] = c
                c+=1
                
                
        
    print(c)
    print("Real label users = ", labels.count(0))
    print("Fake label users = ", labels.count(1))
    print("doc2id = ", len(doc2id))
    row_sum = doc_embeds.sum(1)
    row_sum = list(row_sum)
    c=0
    for s in row_sum:
        if s==0:
            c+=1
    print("Zero entries in final doc_embeds = ", c)

    tsne_file = os.path.join(base_dir, 'tsne', 'doc_tsne.npy')
    print("\nRunning TSNE..")
    doc_embeds = doc_embeds.numpy()
    doc_tsne = TSNE(n_components = config['reduc_dim'], init= 'random', perplexity = 5, learning_rate=75, early_exaggeration= 12).fit_transform(doc_embeds)
    print(doc_tsne.shape)
    np.save(tsne_file, doc_tsne, allow_pickle=True)
    
    if config['reduc_dim']==2:
        fig, ax = plt.subplots()
    elif config['reduc_dim']==3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
    for label in [0, 1]:
        x , y = [], []
        z=[]
        indexer = node2id if config['mode'] == 'gnn' else text2id
        for node, id in indexer.items():
            if node in doc2labels_test:
                if doc2labels_test[str(node)] == label:
                    x.append(doc_tsne[doc2id[str(node)], 0])
                    y.append(doc_tsne[doc2id[str(node)], 1])
                    if config['reduc_dim']==3:
                        z.append(doc_tsne[doc2id[str(node)], 2])
        
        color = 'black' if label==0 else 'red'
        legend = 'Real' if label==0 else 'Fake'
        
        if config['reduc_dim'] ==3:
            ax.scatter(x, y, z, c=color, label=legend,  alpha=0.8)
        else:
            ax.scatter(x, y, c=color, label=legend,  alpha=0.8)
                    
    ax.legend(loc=2, prop={'size': 20})
    plt.show() 
    


def plot_tsne_docs_veracity(config):
            
    base_dir = os.path.join(os.getcwd(), '..', 'data', 'complete_data', 'pheme_cv', 'fold_1')
    doc_splits_file = os.path.join(base_dir, 'doc_splits_filtered.json')
    doc2labels_file = os.path.join(base_dir, 'doc2labels_filtered.json')
    gnn_doc_embeds_file = os.path.join(base_dir, 'doc_embeds_graph_lr_filtered_test_3_{}.pt'.format(config['model_name']))
    text_doc_embeds_file = os.path.join(base_dir, 'doc_embeds_roberta_lr_filtered_21_test.pt')
    text2id_file = os.path.join(base_dir, 'doc2id_encoder_filtered.json')
    node2id_file = os.path.join(base_dir, 'node2id_lr_filtered.json')
    node2id = json.load(open(node2id_file, 'r'))
    text2id = json.load(open(text2id_file, 'r'))
    doc_splits = json.load(open(doc_splits_file, 'r'))
    doc2labels_test = json.load(open(doc2labels_file, 'r'))
    gnn_doc_embeds_loaded = torch.load(gnn_doc_embeds_file)
    text_doc_embeds_loaded = torch.load(text_doc_embeds_file, map_location=torch.device('cpu'))
    
    print(gnn_doc_embeds_loaded.shape, text_doc_embeds_loaded.shape)
    
    row_sum = gnn_doc_embeds_loaded.sum(1)
    row_sum = list(row_sum)
    c=0
    for s in row_sum:
        if s==0:
            c+=1
    print("Zero entries gnn embeds = ", c)
    row_sum = text_doc_embeds_loaded.sum(1)
    row_sum = list(row_sum)
    c=0
    for s in row_sum:
        if s==0:
            c+=1
    print("Zero entries text embeds = ", c)
    
    
    # doc2labels_test = {}
    # for doc,idx in text2id.items():
    #     if str(doc) in doc_splits['test_docs']:
    #         doc2labels_test[str(doc)] = doc2labels[str(doc)]
    
    if config['mode'] == 'gnn':
        doc_embeds = torch.zeros(len(doc2labels_test), gnn_doc_embeds_loaded.shape[1])
        labels = []
        doc2id = {}
        c=0
        for node, id in node2id.items():
            if node in doc2labels_test:
                labels.append(doc2labels_test[str(node)])
                doc_embeds[c, :] = gnn_doc_embeds_loaded[id, :]
                doc2id[str(node)] = c
                c+=1
        
        
        
    if config['mode'] == 'text':
        doc_embeds = torch.zeros(len(doc2labels_test), text_doc_embeds_loaded.shape[1])
        print(doc_embeds.shape)
        labels = []
        doc2id = {}
        c=0
        for node, id in text2id.items():
            if node in doc2labels_test:
                labels.append(doc2labels_test[str(node)])
                doc_embeds[c, :] = text_doc_embeds_loaded[id, :]
                doc2id[str(node)] = c
                c+=1
                
    
    
    if config['mode'] == 'gnn+text':
        doc_embeds = torch.zeros(len(doc2labels_test), text_doc_embeds_loaded.shape[1] + gnn_doc_embeds_loaded.shape[1])
        print(doc_embeds.shape)
        labels = []
        doc2id = {}
        c=0
        for node, id in text2id.items():
            if node in doc2labels_test and node in node2id and str(node) in doc_splits['test_docs']:
                labels.append(doc2labels_test[str(node)])
                gnn_embeds = gnn_doc_embeds_loaded[node2id[str(node)], :]
                text_embeds = text_doc_embeds_loaded[id, :]
                doc_embeds[c, :] = torch.cat((gnn_embeds.unsqueeze(1), text_embeds.unsqueeze(1))).squeeze(1)
                doc2id[str(node)] = c
                c+=1
             
        
    print(c)
    print("Real label users = ", labels.count(0))
    print("Fake label users = ", labels.count(1))
    print("doc2id = ", len(doc2id))
    row_sum = doc_embeds.sum(1)
    row_sum = list(row_sum)
    c=0
    for s in row_sum:
        if s==0:
            c+=1
    print("Zero entries in final doc_embeds = ", c)
        
    
    tsne_file = os.path.join(base_dir, 'tsne', 'doc_tsne.npy')
    print("\nRunning TSNE..")
    doc_embeds = doc_embeds.numpy()
    doc_tsne = TSNE(n_components = config['reduc_dim'], init= 'random', perplexity = 5, learning_rate=50, early_exaggeration= 12).fit_transform(doc_embeds)
    print(doc_tsne.shape)
    np.save(tsne_file, doc_tsne, allow_pickle=True)
    
    if config['reduc_dim']==2:
        fig, ax = plt.subplots()
    elif config['reduc_dim']==3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
    for label in [0, 1]:
        x , y = [], []
        z=[]
        indexer = node2id if config['mode'] == 'gnn' else text2id
        for node, id in indexer.items():
            if node in doc2labels_test:
                if doc2labels_test[str(node)] == label:
                    x.append(doc_tsne[doc2id[str(node)], 0])
                    y.append(doc_tsne[doc2id[str(node)], 1])
                    if config['reduc_dim']==3:
                        z.append(doc_tsne[doc2id[str(node)], 2])
        
        color = 'black' if label==0 else 'red'
        legend = 'Real' if label==0 else 'Fake'
        
        if config['reduc_dim'] ==3:
            ax.scatter(x, y, z, c=color, label=legend,  alpha=0.8)
        else:
            ax.scatter(x, y, c=color, label=legend,  alpha=0.8)
                    
    ax.legend(loc=2, prop={'size': 20})
    plt.show() 

def plot_relative_density():
    
    width = 0.45
    gat = [79.49, 78.5, 76.1]
    gcn = [77.9, 76.59, 76.2]
    sage = [76.08, 75.11, 76]
    rgcn = [85.22, 83.5, 82.2]
    
    print(plt.style.available)
    plt.style.use('seaborn-talk')
    fig, ax = plt.subplots()
    
    plt.title("Model Performances over varying Graph density", fontsize = 23)
    # plt.figure(figsize=(10, 10))
    densities = [1.0, 0.76, 0.65]
    ind = np.arange(3)
    plt.ylim(0.4, 1.0)
    ax.bar(ind, densities, width, alpha = 0.3, label='Relative Density')
    plt.ylabel('Relative Graph Density', fontsize= 17)
    plt.xlabel('Subset of users', fontsize= 23)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(['Original', 'excl. >10%', 'excl. >1%'], fontsize = 15)
    
    axes2 = plt.twinx()
    axes2.plot(ind, gat, marker='o', color='steelblue',linewidth=3, label='GAT')
    axes2.plot(ind, gcn, marker='o', color='mediumseagreen', linewidth=3, label='GCN')
    axes2.plot(ind, sage, marker='o', color='indianred', linewidth=3, label='SAGE')
    axes2.plot(ind, rgcn, marker='o', color='black', linewidth=3, label='R-GCN')
    axes2.set_ylim(60, 90)
    axes2.set_ylabel('RoBERTa+GNN (F1)', rotation=270, labelpad=20, fontsize= 17)
    axes2.legend(loc=1, prop={'size': 15})
    plt.show()
    return None




def plot_sharing_behav():
    width=0.2
    N=2
    real = [1251, 2278]
    fake = [11389, 1462]
    both = [17322, 10936]  
    denoms= [29962, 14676]
    
    real = [r/d for r,d in zip(real, denoms)]
    fake = [r/d for r,d in zip(fake, denoms)]
    both = [r/d for r,d in zip(both, denoms)]
    # ind = np.arange(N)    # the x locations for the groups
    ind = [0, 0.75]
    print(real)
    print(fake)
    print(both)
    
    print(plt.style.available)
    plt.style.use('seaborn-paper')
    
    p1 = plt.bar(ind, both, width, color='lightskyblue', edgecolor='white', align = 'center', alpha=1)
    p2 = plt.bar(ind, real, width, bottom=both, color='mediumseagreen', edgecolor='white', align = 'center', alpha=0.9)
    p3 = plt.bar(ind, fake, width, bottom = np.array(both)+ np.array(real), color='indianred', edgecolor='white', align = 'center', alpha=0.75)
    
    plt.ylabel('Ratio of user-types', fontsize = 12)
    plt.title('Sharing behavior of different types of users', fontsize = 12)
    plt.xticks(ind, ('Gossipcop', 'HealthStory'), fontsize = 12)
    # plt.ylim(0.2,1)
    # plt.yticks(np.arange(0, 81, 10))
    plt.legend((p1[0], p2[0], p3[0]), ('Both', 'Real', 'Fake'), prop={'size': 10})
    
    plt.show()
    
    return None    



def plot_common_users_sharing():
    
    N = 2
    fakes = [8.3, 3.9]
    reals = [35.7, 10.6]

    ind = np.arange(N)    # the x locations for the groups
    width = 0.2        # the width of the bars
    p1 = plt.bar(ind, reals, width, color='mediumseagreen', edgecolor='white', align = 'center', alpha=0.9)
    p2 = plt.bar(ind+width, fakes, width, color='indianred', edgecolor='white', align = 'center', alpha=0.75)

    
    plt.ylabel('Avg. shares per class', fontsize = 12)
    plt.title('Sharing behavior of Users that share both classes', fontsize = 12)
    plt.xticks(ind, ('Gossipcop', 'HealthStory'), fontsize = 12)
    # plt.yticks(np.arange(0, 81, 10))
    plt.legend((p1[0], p2[0]), ('Real','Fake'), prop={'size': 15})    
    plt.show()
    
    

def plot_optimum_threshold():
    
    N = 6
    val = [44.38, 49.14, 54.32, 56.45, 50.19, 44.35]
    test = [51.24, 59.15, 59.87, 61.24, 62.16, 54.06]

    ind = np.arange(N)    # the x locations for the groups
    # p1 = plt.bar(ind, reals, width, color='mediumseagreen', edgecolor='white', align = 'center', alpha=0.9)
    # p2 = plt.bar(ind+width, fakes, width, color='indianred', edgecolor='white', align = 'center', alpha=0.75)
    
    plt.plot(ind, val, marker='o', color='mediumseagreen', linewidth=3, label='val-F1')
    plt.plot(ind, test, marker='o', color='indianred', linestyle= 'dashed', linewidth=3, label='test-F1')

    
    plt.ylabel('F1(fake) per split', fontsize = 15)
    plt.xlabel('Top N user subsets', fontsize = 15)
    plt.title('Optimum threshold for top N users', fontsize = 15)
    plt.xticks(ind, ('All', 'Top60k', 'Top40k', 'Top20k', 'Top8k', 'Top6k'), fontsize = 12)
    # plt.yticks(np.arange(0, 81, 10))
    plt.legend(loc=2, prop={'size': 15})    
    plt.show()
    

def plot_event_user_overlap():
    print("\n\n" + "-"*100 + "\n \t\t   Checking overlapping users for PHEME\n" + '-'*100)  
    data_dir = './data'
    # Creating the CV folds from the remaining events
    events = ['ch', 'ebola', 'ferg', 'german', 'gurlitt', 'ottawa', 'putin', 'sydney', 'toronto']
    src_dir = os.path.join(data_dir, 'base_data', 'pheme_cv')
    src_users_list =  defaultdict(set)
    event_follower_list = defaultdict(set)
    user_contexts = ['user_followers_filtered', 'user_following_filtered']
    for fold, event in enumerate(events):
        print("\nCreating fold_{}  with  {}  as test set\n".format(fold+1, event) + "-"*50 )
        test_data_dir = os.path.join(src_dir, event)
        c=0
        for root, dirs, files in os.walk(test_data_dir):
            for file in files:
                if file.startswith('.') or file.startswith('structure') or file.startswith('annotation') or root.endswith('reactions'):
                    continue
                else:
                    src_file_path = os.path.join(root, file)
                    src_file = json.load(open(src_file_path, 'r'))
                    user_id = int(src_file['user']['id'])
                    src_users_list[event].update([user_id]) 
                    for user_context in user_contexts:
                        src_file_path = os.path.join(data_dir, 'complete_data', 'pheme', user_context, str(user_id)+'.json')
                        # user_id = src_file_path.split(".")[0]
                        src_file = json.load(open(src_file_path, 'r'))
                        followers = src_file['followers'] if user_context == 'user_followers_filtered' else src_file['following']   
                        followers = list(map(int, followers))
                        event_follower_list[event].update(followers)
                    c+=1
                    if c%2000 == 0:
                        print("{} done...".format(c))
        print("Total = ", c)
        print("Followers = ", len(event_follower_list[event]))
        print("Users = ", len(src_users_list[event]))
    all_users = set()
    for event, user in src_users_list.items():
        print("No. of src_users_list in {} = {}".format(event, len(user)))
        all_users.update(user)
    all_users = list(all_users)
    
    
    
    # Fold-wise heatmap
    heatmap = np.zeros((len(events), 1))
    for i, event in enumerate(events):
        # event_users = src_users_list[event]
        event_users = event_follower_list[event]
        event_users.update(list(src_users_list[event]))
        rest_users = set()
        # rest_events = events.copy()
        # print(len(rest_events))
        # rest_events.remove(event)
        # print(len(rest_events))
        
        for rest in events: #rest_events:
            if rest == event:
                continue
            rest_users.update(src_users_list[rest])
        
        print(len(rest_users), len(event_users))
        common = event_users.intersection(rest_users)
        print(len(common))
        # heatmap[i] = len(common)/len(rest_users)
        heatmap[i] = len(common)/len(event_users)
    
    # fig2= plt.figure(2)
    # plt.figure(figsize = (5,20))
    # Plot the degrees as Heatmap
    bx = sns.heatmap(heatmap, linewidth=0.5, cmap="YlGnBu", annot=True, annot_kws={"size": 16})
    for i in range(0,10):
        bx.axhline(i, color='white', lw=15)
    bx.set_xticklabels([''])
    bx.set_yticklabels(events, rotation=0)
    bx.set_xlabel("Train-Test Overlap")
    bx.set_ylabel("Test events")
    # bx.set_title("Overlap between source-authors across different CV folds", fontsize=20)
    bx.set_title("Overlap between source-tweet users", fontsize=20)
    plt.show()
    
   
    
    # # Event-wise heatmap (avg degree in/out)
    # heatmap = np.zeros((len(events), len(events)))
    # norm = []
    # for i in range(len(events)):
    #     for j in range(len(events)):
    #         cmn=0
    #         # a = set(event_follower_list[events[i]])
    #         a = set(src_users_list[events[i]])
    #         b = set(src_users_list[events[j]])
    #         # if i==j:
    #         #     norm.append(len(a.intersection(b)))
    #         for x in a:
    #             if x in b:
    #                 cmn+=1
    #         print("Overlap of src_users_list between {} and {} =  {}".format(events[i], events[j], len(a.intersection(b))))
    #         # heatmap[i,j] = len(a.intersection(b))/len(src_users_list[events[i]])
    #         if i==j:
    #             heatmap[i,j] = 0
    #         else:
    #             # heatmap[i,j] = cmn/len(event_follower_list[events[i]])
    #             heatmap[i,j] = cmn/len(src_users_list[events[i]])
    

    # # Plot the degrees as Heatmap
    # print(heatmap.shape)
    # ax = sns.heatmap(heatmap, linewidth=0.5,cmap="YlGnBu", annot=True, annot_kws={"size": 12})
    # # ax = sns.heatmap(heatmap, linewidth=0.5, cmap="Blues", annot=True, annot_kws={"size": 12})
    # for i in range(0,10):
    #     ax.axhline(i, color='white', lw=20)
    # # ax.set_xlabel("Avg out-degree of the event")
    # # ax.set_ylabel("Avg in-degree of the event")
    # ax.set_xticklabels(events)
    # ax.set_yticklabels(events, rotation=0)
    # ax.set_title("Overlap between source users across different events", fontsize=20)
    # plt.show()
    
    return None    



def plot_full_pheme(fold=1):
    
    x_file = './data/complete_data/pheme_cv/fold_{}/feat_matrix_just_lr.npz'.format(fold)
    y_file = './data/complete_data/pheme_cv/fold_{}/all_labels_just.json'.format(fold)
    edge_index_file = './data/complete_data/pheme_cv/fold_{}/adj_matrix_just_lr_edge.npy'.format(fold)
    node_type_file = './data/complete_data/pheme_cv/fold_{}/node_type_lr_train.npy'.format(fold)
    
    
    x_data = load_npz(x_file)
    # x_data = np.load(x_file)
    x_data = torch.from_numpy(x_data.toarray())
    y_data = json.load(open(y_file, 'r'))
    y_data = torch.LongTensor(y_data['all_labels'])
    edge_index_data = np.load(edge_index_file)
    edge_index_data = torch.from_numpy(edge_index_data)
    node_type = np.load(node_type_file)
    y=None
    if y is None:
           y = torch.zeros(edge_index_data.max().item() + 1,
                           device=edge_index_data.device)
    data = Data(x=x_data.float(), edge_index = edge_index_data.long(), edge_attr=node_type, y=node_type).to('cpu')
    
    G = to_networkx(data, node_attrs=['y'], edge_attrs=None)
    # mapping = {k: i for k, i in enumerate(subset.tolist())}
    # G = nx.relabel_nodes(G, mapping)

    # kwargs['with_labels'] = kwargs.get('with_labels') or True
    kwargs = {}
    kwargs['with_labels'] = False
    kwargs['font_size'] = kwargs.get('font_size') or 5
    kwargs['node_size'] = kwargs.get('node_size') or 800
    kwargs['cmap'] = kwargs.get('cmap') or 'RdYlGn'

    pos = nx.spring_layout(G)
    ax = plt.gca()
    for source, target, data in G.edges(data=True):
        ax.annotate(
            '', xy=pos[target], xycoords='data', xytext=pos[source],
            textcoords='data', arrowprops=dict(
                arrowstyle="->",
                alpha=0.5,
                shrinkA=sqrt(kwargs['node_size']) / 2.0,
                shrinkB=sqrt(kwargs['node_size']) / 2.0,
                connectionstyle="arc3,rad=0.1",
            ))
    nx.draw_networkx_nodes(G, pos, node_color=y.tolist(), alpha = 0.5, **kwargs)
    nx.draw_networkx_labels(G, pos, **kwargs)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Arguments for TSNE plots
    parser.add_argument('--base_dir', type = str, default = '../data/complete_data/HealthStory',
                          help='path to the base dataset folder')
    parser.add_argument('--model_name', type = str, default = 'gat',
                          help='which GNN model to plot: gcn / gat / graph_sage')
    parser.add_argument('--mode', type = str, default = 'gnn+text',
                          help='which embeddings to plot: gnn / text / gnn+text')
    parser.add_argument('--reduc_dim', type = int, default = 3,
                          help='How many dimensions to reduce the tSNE to')
    
    
    args, unparsed = parser.parse_known_args()
    config = args.__dict__

    # plot_relative_density()
    # plot_sharing_behav()
    # plot_common_users_sharing()
    # plot_optimum_threshold()
    # plot_event_user_overlap()
    # plot_full_pheme()
    
    # plot_tsne_docs_news(config)
    # plot_tsne_docs_health(config)
    plot_tsne_docs_veracity(config)