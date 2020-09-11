import os, time, json, sys, gc
import numpy as np
import nltk, pandas
import torch
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, classification_report
from scipy.sparse import csr_matrix, lil_matrix, save_npz, load_npz
from torch_geometric.data import Data, DataLoader, GraphSAINTRandomWalkSampler, GraphSAINTNodeSampler, GraphSAINTEdgeSampler
from torch_geometric.data.cluster import ClusterData, ClusterLoader
from torch_geometric.utils import remove_isolated_nodes, contains_isolated_nodes, contains_self_loops

import warnings
warnings.filterwarnings("ignore")
sys.path.append("..")
from utils.data_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Hacky trick to avoid the MAXSIZE python error
# import csv
# maxInt = sys.maxsize
# while True:
#     # decrease the maxInt value by factor 2 as long as the OverflowError occurs.
#     try:
#         csv.field_size_limit(maxInt)
#         break
#     except OverflowError:
#         maxInt = int(maxInt/10)


# For printing cleaner numpy arrays
float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})


def prepare_lr_training(config, seed, fold=None):
    print("="*100 + "\n\t\t\t\t Preparing LR Data for {} mode\n".format(config['mode']) + "="*100)
    
    train_dataset = LR_Dataset(config, split='train', seed=seed) if config['data_name'] != 'pheme' else LR_Dataset_pheme(config, split='train', seed=seed, fold=fold)
    val_dataset = LR_Dataset(config, split='val', seed=seed) if config['data_name'] != 'pheme' else LR_Dataset_pheme(config, split='test', seed=seed, fold=fold)
    test_dataset = LR_Dataset(config, split='test', seed=seed) if config['data_name'] != 'pheme' else LR_Dataset_pheme(config, split='test', seed=seed, fold=fold)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config['batch_size'], collate_fn=collate_for_lr, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=config['batch_size'], collate_fn=collate_for_lr, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=config['batch_size'], collate_fn=collate_for_lr, shuffle=True)
        
    return train_loader, val_loader, test_loader
        



def prepare_gnn_training_mtl(config, fold=None, verbose=True):
    
    if config['data_name'] == 'pheme':
        if verbose:
            print("="*100 + "\n\t\t\t\t Preparing Data for fold {}\n".format(fold) + "="*100)
        start = time.time()
        
        print("\n\n==>> Loading feature matrix and adj matrix....")
        # x = [num_nodes, node_feats]
        # edge_index = [2, num_edges]
        x_file = os.path.join(config['data_path'], 'pheme_cv', 'fold_{}'.format(fold), 'feat_matrix_lr_mtl.npz')
        y_file_stance = os.path.join(config['data_path'], 'pheme_cv', 'fold_{}'.format(fold), 'all_labels_stance_mtl.json')
        y_file_veracity = os.path.join(config['data_path'], 'pheme_cv', 'fold_{}'.format(fold), 'all_labels_veracity_mtl.json')
        edge_index_file = os.path.join(config['data_path'], 'pheme_cv', 'fold_{}'.format(fold), 'adj_matrix_lr_edge_mtl.npy')
        node_type_file = os.path.join(config['data_path'], 'pheme_cv', 'fold_{}'.format(fold), 'node_type_lr_train.npy')
        if config['model_name'] in ['rgcn', 'rgat', 'rsage']:
            edge_type_file = os.path.join(config['data_path'], 'pheme_cv', 'fold_{}'.format(fold), 'edge_type_lr_edge_mtl.npy')
        node2id_file = os.path.join(config['data_path'], 'pheme_cv', 'fold_{}'.format(fold), 'node2id_lr_mtl.json')
        split_masks = json.load(open(os.path.join(config['data_path'], 'pheme_cv', 'fold_{}'.format(fold), 'split_mask_mtl.json'), 'r'))
        
        x_data = load_npz(x_file)
        x_data = torch.from_numpy(x_data.toarray())
        y_data_stance = json.load(open(y_file_stance, 'r'))
        y_data_veracity = json.load(open(y_file_veracity, 'r'))
        y_data_stance = torch.LongTensor(y_data_stance['all_labels'])
        y_data_veracity = torch.LongTensor(y_data_veracity['all_labels'])
        edge_index_data = np.load(edge_index_file)
        edge_index_data = torch.from_numpy(edge_index_data)
        if config['model_name'] in ['rgcn', 'rgat', 'rsage']:
            edge_type_data = np.load(edge_type_file)
            edge_type_data = torch.from_numpy(edge_type_data).long()
        else:
            edge_type_data = None
        node2id = json.load(open(node2id_file, 'r'))
        node_type = np.load(node_type_file)
        node_type = torch.from_numpy(node_type).float()
        
        num_nodes, vocab_size = x_data.shape
        isolated_nodes = contains_isolated_nodes(edge_index= edge_index_data)
        self_loops = contains_self_loops(edge_index= edge_index_data)
        
        print("\n\n==>> Clustering the graph and preparing dataloader....")
        data_stance = Data(x=x_data.float(), edge_index = edge_index_data.long(), edge_attr=edge_type_data, y=y_data_stance)
        data_veracity = Data(x=x_data.float(), edge_index = edge_index_data.long(), edge_attr=edge_type_data, y=y_data_veracity)
        del x_data
        del y_data_stance
        del y_data_veracity
        del edge_index_data
        
        new_num_nodes, _ = data_veracity.x.shape
        num_edges = data_veracity.num_edges
        
        
        data_stance.train_mask_stance = torch.FloatTensor(split_masks['train_mask_stance'])
        data_veracity.train_mask_veracity = torch.FloatTensor(split_masks['train_mask_veracity'])
        data_stance.test_mask_stance = torch.FloatTensor(split_masks['test_mask_stance'])
        data_veracity.test_mask_veracity = torch.FloatTensor(split_masks['test_mask_veracity'])
        data_stance.representation_mask_stance = torch.FloatTensor(split_masks['representation_mask_stance'])
        data_veracity.representation_mask_veracity = torch.FloatTensor(split_masks['representation_mask_veracity'])
        data_veracity.node2id = torch.tensor(list(node2id.values()))
        data_stance.node2id = torch.tensor(list(node2id.values()))
        data_veracity.node_type = node_type
        
        if config['cluster']:
            cluster_data_stance = ClusterData(data_stance, num_parts=config['clusters'], recursive=False)
            # del data_stance
            loader_stance = ClusterLoader(cluster_data_stance, batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=0)
            del cluster_data_stance
            
            cluster_data_veracity = ClusterData(data_veracity, num_parts=config['clusters'], recursive=False)
            # del data_veracity
            loader_veracity = ClusterLoader(cluster_data_veracity, batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=0)
            del cluster_data_veracity
        else:
            loader_stance, loader_veracity = None, None
            
    
    if verbose:
        print("\n\n" + "-"*50 + "\nDATA STATISTICS:\n" + "-"*50)
        print("Contains isolated nodes = ", isolated_nodes)
        print("Contains self loops = ", self_loops)
        print("Vocabulary size = ", vocab_size)
        print('No. of nodes in graph = ', num_nodes)
        # print('No. of nodes after removing isolated nodes = ', new_num_nodes)
        print("No. of edges in graph = ", num_edges)
        
        print("\nNo.of train instances (veracity) = ", data_veracity.train_mask_veracity.sum().item())
        print("No.of test instances (veracity) = ", data_veracity.test_mask_veracity.sum().item())
        print("No.of train instances (stance) = ", data_stance.train_mask_stance.sum().item())
        print("No.of test instances (stance) = ", data_stance.test_mask_stance.sum().item())
            
        
        end = time.time()
        hours, minutes, seconds = calc_elapsed_time(start, end)
        print("\n"+ "-"*50 + "\nTook  {:0>2} hours: {:0>2} mins: {:05.2f} secs  to Prepare Data\n".format(hours,minutes,seconds))
    
    return loader_stance, loader_veracity, vocab_size , data_stance, data_veracity





def prepare_gnn_training(config, fold=None, verbose=True):
    
    if config['data_name'] == 'pheme':
        print("="*100 + "\n\t\t\t\t Preparing Data for fold {}\n".format(fold) + "="*100)
        start = time.time()
        
        print("\n\n==>> Loading feature matrix and adj matrix....")
        # x = num_nodes X node_feats
        # edge_index = 2 X num_edges
        x_file = os.path.join(config['data_path'], 'pheme_cv', 'fold_{}'.format(fold), 'feat_matrix_just_lr.npz')
        y_file = os.path.join(config['data_path'], 'pheme_cv', 'fold_{}'.format(fold), 'all_labels_just.json')
        edge_index_file = os.path.join(config['data_path'], 'pheme_cv', 'fold_{}'.format(fold), 'adj_matrix_just_lr_edge.npy') 
        node_type_file = os.path.join(config['data_path'], 'pheme_cv', 'fold_{}'.format(fold), 'node_type_lr_train.npy')
        if config['model_name'] in ['rgcn', 'rgat', 'rsage']:
            edge_type_file = os.path.join(config['data_path'], 'pheme_cv', 'fold_{}'.format(fold), 'edge_type_just_lr_edge.npy') 
        node2id_file = os.path.join(config['data_path'], 'pheme_cv', 'fold_{}'.format(fold), 'node2id_lr_just.json')
        split_masks = json.load(open(os.path.join(config['data_path'], 'pheme_cv', 'fold_{}'.format(fold), 'split_mask_just.json'), 'r'))
        
        x_data = load_npz(x_file)
        x_data = torch.from_numpy(x_data.toarray())
        y_data = json.load(open(y_file, 'r'))
        y_data = torch.LongTensor(y_data['all_labels'])
        edge_index_data = np.load(edge_index_file)
        edge_index_data = torch.from_numpy(edge_index_data)
        if config['model_name'] in ['rgcn', 'rgat', 'rsage']:
            edge_type_data = np.load(edge_type_file)
            edge_type_data = torch.from_numpy(edge_type_data).long()
        else:
            edge_type_data = None
        node2id = json.load(open(node2id_file, 'r'))
        node_type = np.load(node_type_file)
        # node_type = torch.from_numpy(node_type).float()
        
        num_nodes, vocab_size = x_data.shape
        isolated_nodes = contains_isolated_nodes(edge_index= edge_index_data)
        self_loops = contains_self_loops(edge_index= edge_index_data)
        
        print("\n\n==>> Clustering the graph and preparing dataloader....")
        data = Data(x=x_data.float(), edge_index = edge_index_data.long(), edge_attr=edge_type_data, y=y_data)
        
        new_num_nodes, _ = data.x.shape
        
        
        data.train_mask = torch.FloatTensor(split_masks['train_mask'])
        data.val_mask = torch.FloatTensor(split_masks['test_mask'])
        data.representation_mask = torch.FloatTensor(split_masks['repr_mask'])
        data.node2id = torch.tensor(list(node2id.values()))
        # data.node_type = node_type
        
        
        if config['cluster']:
            cluster_data = ClusterData(data, num_parts=config['clusters'], recursive=False)
            loader = ClusterLoader(cluster_data, batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=0)
        else:
            loader=None
            
    else:
        if verbose:
            print("="*100 + "\n\t\t\t\t Preparing Data for {}\n".format(config['data_name']) + "="*100)
        start = time.time()
        
        if verbose:
            print("\n\n==>> Loading feature matrix and adj matrix....") 
            
        if config['data_name'] in ['gossipcop', 'politifact']:
            x_file = os.path.join(config['data_path'], config['data_name'], 'feat_matrix_lr_train_30_5.npz'.format(config['data_name']))
            y_file = os.path.join(config['data_path'], config['data_name'], 'all_labels_lr_train_30_5.json'.format(config['data_name']))
            # adj_name = 'adj_matrix_lr_train_30_5_edge.npy'.format(config['data_name']) if config['model_name'] != 'HGCN' else 'adj_matrix_lr_train_30_5.npz'.format(config['data_name'])
            adj_name = 'adj_matrix_lr_train_30_5_edge.npy'.format(config['data_name'])
            edge_index_file = os.path.join(config['data_path'], config['data_name'], adj_name)
            node2id_file = os.path.join(config['data_path'], config['data_name'], 'node2id_lr_train_30_5.json'.format(config['data_name']))
            node_type_file = os.path.join(config['data_path'], config['data_name'], 'node_type_lr_train_30_5.npy'.format(config['data_name']))
            if config['model_name'] in ['rgcn', 'rgat', 'rsage']:
                edge_type_file = os.path.join(config['data_path'], config['data_name'], 'edge_type_lr_train_30_5_edge.npy'.format(config['data_name']))
        else:
            x_file = os.path.join(config['data_path'], config['data_name'], 'feat_matrix_lr_top10_train.npz')
            y_file = os.path.join(config['data_path'], config['data_name'], 'all_labels_lr_top10_train.json')
            # adj_name = 'adj_matrix_lr_top10_train_edge.npy' if config['model_name'] != 'HGCN' else 'adj_matrix_lr_top10_train.npz'
            adj_name = 'adj_matrix_lr_top10_train_edge.npy'
            edge_index_file = os.path.join(config['data_path'], config['data_name'], adj_name)
            node2id_file = os.path.join(config['data_path'], config['data_name'], 'node2id_lr_top10_train.json')
            node_type_file = os.path.join(config['data_path'], config['data_name'], 'node_type_lr_top10_train.npy')
            if config['model_name'] in ['rgcn', 'rgat', 'rsage']:
                edge_type_file = os.path.join(config['data_path'], config['data_name'], 'edge_type_lr_top10_edge.npy')
        
        
        # if config['model_name'] != 'HGCN':
        #     edge_index_data = np.load(edge_index_file)
        #     edge_index_data = torch.from_numpy(edge_index_data).long()
            
        # elif config['model_name'] == 'HGCN':
        #     edge_index_data = load_npz(edge_index_file)
            
        #     # edge_index_data = torch.from_numpy(edge_index_data.toarray())
            
        #     edge_index_data = edge_index_data.tocoo()
        #     indices = torch.from_numpy(np.vstack((edge_index_data.row, edge_index_data.col)).astype(np.int64))
        #     values = torch.Tensor(edge_index_data.data)
        #     shape = torch.Size(edge_index_data.shape)
        #     edge_index_data = torch.sparse.FloatTensor(indices, values, shape)
        
        edge_index_data = np.load(edge_index_file)
        edge_index_data = torch.from_numpy(edge_index_data).long()
        
        x_data = load_npz(x_file)
        x_data = torch.from_numpy(x_data.toarray())
        y_data = json.load(open(y_file, 'r'))
        y_data = torch.LongTensor(y_data['all_labels'])
        node2id = json.load(open(node2id_file, 'r'))
        # node_type = np.load(node_type_file)
        # node_type = torch.from_numpy(node_type).float()
        if config['model_name'] in ['rgcn', 'rgat', 'rsage']:
            edge_type_data = np.load(edge_type_file)
            edge_type_data = torch.from_numpy(edge_type_data).long()
        else:
            edge_type_data = None
        
        num_nodes, vocab_size = x_data.shape
        if config['model_name'] != 'HGCN':
            isolated_nodes = contains_isolated_nodes(edge_index= edge_index_data)
            self_loops = contains_self_loops(edge_index= edge_index_data)
        
        if verbose:
            print("\n\n==>> Clustering the graph and preparing dataloader....")
            
        data = Data(x=x_data.float(), edge_index = edge_index_data.long(), edge_attr = edge_type_data, y=y_data)
        new_num_nodes, _ = data.x.shape

        
        split_mask_file = os.path.join(config['data_path'], config['data_name'], 'split_mask_lr_30_5.json') if config['data_name'] in ['gossipcop', 'politifact'] else \
            os.path.join(config['data_path'], config['data_name'], 'split_mask_top10.json')
        split_masks = json.load(open(split_mask_file, 'r'))
        data.train_mask = torch.FloatTensor(split_masks['train_mask'])
        data.val_mask = torch.FloatTensor(split_masks['val_mask'])
        # data.representation_mask = torch.FloatTensor(split_masks['repr_mask']) 
        data.node2id = torch.tensor(list(node2id.values()))
        # data.node_type = node_type
            
        
        if not config['full_graph']:
            if config['cluster'] :
                cluster_data = ClusterData(data, num_parts=config['clusters'], recursive=False)
                loader = ClusterLoader(cluster_data, batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=0)
            elif config['saint'] == 'random_walk':
                loader = GraphSAINTRandomWalkSampler(data, batch_size=6000, walk_length=2, num_steps=5, sample_coverage=100, num_workers=0)
            elif config['saint'] == 'node':
                loader = GraphSAINTNodeSampler(data, batch_size=6000, num_steps=5, sample_coverage=100, num_workers=0)
            elif config['saint'] == 'edge':
                loader = GraphSAINTEdgeSampler(data, batch_size=6000, num_steps=5, sample_coverage=100, num_workers=0)
        else:
            loader=None
        
    
    if verbose:
        print("\n\n" + "-"*50 + "\nDATA STATISTICS:\n" + "-"*50)
        if config['model_name'] != 'HGCN':
            print("Contains isolated nodes = ", isolated_nodes)
            print("Contains self loops = ", self_loops)
        print("Vocabulary size = ", vocab_size)
        print('No. of nodes in graph = ', num_nodes)
        print('No. of nodes after removing isolated nodes = ', new_num_nodes)
        print("No. of edges in graph = ", data.num_edges)
        
        print("\nNo.of train instances = ", data.train_mask.sum().item())
        print("No.of val instances = ", data.val_mask.sum().item())
        print("No.of test instances = ", num_nodes - data.train_mask.sum().item() - data.val_mask.sum().item())
            
        
        end = time.time()
        hours, minutes, seconds = calc_elapsed_time(start, end)
        print("\n"+ "-"*50 + "\nTook  {:0>2} hours: {:0>2} mins: {:05.2f} secs  to Prepare Data\n".format(hours,minutes,seconds))
    
    return loader, vocab_size, data
    
    
    
    

def prepare_transformer_training(config, fold=None, verbose=True): 
    if verbose:
        if fold:
            print("="*100 + "\n\t\t\t\t Preparing Data for fold {}\n".format(fold) + "="*100)
        else:
            print("="*100 + "\n\t\t\t\t\t Preparing Data\n" + "="*100)
        
        print("\n\n==>> Loading Data splits and tokenizing each document....")
        
    start = time.time()
    
    if config['data_name'] in ['gossipcop', 'politifact', 'HealthRelease', 'HealthStory']:
        train_data_dir = os.path.join(config['data_path'], 'train.tsv')
        val_data_dir = os.path.join(config['data_path'], 'val.tsv')
        test_data_dir = os.path.join(config['data_path'], 'test.tsv')
    elif config['data_name'] == 'pheme':
        if fold==0:
            train_data_dir = os.path.join(config['data_path'], 'train_1.tsv')
            val_data_dir = os.path.join(config['data_path'], 'test_1.tsv')
            test_data_dir = os.path.join(config['data_path'], 'test_1.tsv')
            # train_data_dir = os.path.join(config['data_path'], 'train_filtered_1.tsv')
            # val_data_dir = os.path.join(config['data_path'], 'test_filtered_1.tsv')
            # test_data_dir = os.path.join(config['data_path'], 'test_filtered_1.tsv')
        else:
            # train_data_dir = os.path.join(config['data_path'], 'train_filtered_{}.tsv'.format(fold))
            # val_data_dir = os.path.join(config['data_path'], 'test_filtered_{}.tsv'.format(fold))
            # test_data_dir = os.path.join(config['data_path'], 'test_filtered_{}.tsv'.format(fold))
            train_data_dir = os.path.join(config['data_path'], 'train_{}.tsv'.format(fold))
            val_data_dir = os.path.join(config['data_path'], 'test_{}.tsv'.format(fold))
            test_data_dir = os.path.join(config['data_path'], 'test_{}.tsv'.format(fold))
                
        
    directories = ['train_data_dir', 'val_data_dir', 'test_data_dir']
    train_data, val_data, test_data, train_labels, val_labels, test_labels = [], [], [], [], [], []
    
    for i in range(len(directories)):
        data_dir = eval(directories[i])
        df = pandas.read_csv(data_dir, sep='\t', encoding = 'utf8', header=None)
        # Each ele is a document, that we tokenize
        if i==0:
            train_df = pd.DataFrame({'text': df[0].replace(r'\n', ' ', regex=True),'label': df[1]})
            # print(train_df.head())
   
        elif i==1:
            val_df = pd.DataFrame({'text': df[0].replace(r'\n', ' ', regex=True),'label': df[1]})
            # print(val_df.head())
        else:
            test_df = pd.DataFrame({'text': df[0].replace(r'\n', ' ', regex=True),'label': df[1]})
            # print(test_df.head())
    if verbose:
        print("\n\n" + "-"*50 + "\nDATA STATISTICS:\n" + "-"*50)
        print('No. of target classes = ', config["n_classes"])
        print('No. of train instances = ', len(train_df['label']))
        print('No. of dev instances = ', len(val_df['label']))
        print('No. of test instances = ', len(test_df['label']))
    
        end = time.time()
        hours, minutes, seconds = calc_elapsed_time(start, end)
        print("\n"+ "-"*50 + "\nTook  {:0>2} hours: {:0>2} mins: {:05.2f} secs  to Prepare Data\n".format(hours,minutes,seconds))           
    return train_df, val_df, test_df



def prepare_HAN_elmo_training(config):
    print("="*100 + "\n\t\t\t\t\t Preparing Data\n" + "="*100)
    start = time.time()   
    # Getting Data Splits: train, dev, test and their ids for ELmo
    print("\n\n==>> Loading Data splits and tokenizing each document....")   
    train_data, val_data, test_data = Dataset_Helper_HAN.get_dataset(config, lowercase_sentences=config['lowercase'])
    
    # Getting iterators for each set
    print("\n==>> Preparing Iterators....")
    train_loader, val_loader, test_loader = DataLoader_Helper_HAN.create_dataloaders(
        train_dataset=train_data,
        validation_dataset=val_data,
        test_dataset=test_data,
        batch_size=config['batch_size'],
        shuffle=True)    
    print("\n\n" + "-"*50 + "\nDATA STATISTICS:\n" + "-"*50)
    print('No. of target classes = ', 2)
    print('No. of train instances = ', len(train_loader.dataset))
    print('No. of dev instances = ', len(val_loader.dataset))
    print('No. of test instances = ', len(test_loader.dataset))   
    end = time.time()
    hours, minutes, seconds = calc_elapsed_time(start, end)
    print("\n"+ "-"*50 + "\nTook  {:0>2} hours: {:0>2} mins: {:05.2f} secs  to Prepare Data\n".format(hours,minutes,seconds))   
    return train_loader, val_loader, test_loader



# def prepare_glove_baseline_training(config, fold=None):
#     if fold:
#         print("="*100 + "\n\t\t\t\t Preparing Data for fold {}\n".format(fold) + "="*100)
#     else:
#         print("="*100 + "\n\t\t\t\t\t Preparing Data\n" + "="*100)
        
#     start = time.time()
#     if config['data_name'] in ['politifact', 'gossipcop']:
#         if config['model_name'] == 'han':
#             TEXT, LABEL, train_batch_loader, dev_batch_loader, test_batch_loader, train_split, val_split, test_split = FakeNews_HAN.main_handler(config, config['data_path'], shuffle=True)
#         else:
#             TEXT, LABEL, train_batch_loader, dev_batch_loader, test_batch_loader, train_split, val_split, test_split = FakeNews_baseline.main_handler(config, config['data_path'], shuffle=True)
    
#     elif config['data_name'] in ['HealthStory', 'HealthRelease']:
#         if config['model_name'] == 'han':
#             TEXT, LABEL, train_batch_loader, dev_batch_loader, test_batch_loader, train_split, val_split, test_split = FakeHealth_HAN.main_handler(config, config['data_path'], fold, shuffle=True)
#         else:
#             TEXT, LABEL, train_batch_loader, dev_batch_loader, test_batch_loader, train_split, val_split, test_split = FakeHealth.main_handler(config, config['data_path'], fold, shuffle=True)
            
#     elif config['data_name'] == 'pheme':
#         if config['model_name'] == 'han':
#             TEXT, LABEL, train_batch_loader, dev_batch_loader, test_batch_loader, train_split, val_split, test_split = Rumor_HAN.main_handler(config, config['data_path'], fold, shuffle=True)
#         else:
#             TEXT, LABEL, train_batch_loader, dev_batch_loader, test_batch_loader, train_split, val_split, test_split = Rumor.main_handler(config, config['data_path'], fold, shuffle=True)
    
    
#     vocab_size = len(TEXT.vocab)
#     config['vocab_size'] = vocab_size
#     print("\n\n" + "-"*50 + "\nDATA STATISTICS:\n" + "-"*50)
#     print("\nVocabulary size = ", vocab_size)
#     print('No. of target classes = ', train_batch_loader.dataset.NUM_CLASSES)
#     print('No. of train instances = ', len(train_batch_loader.dataset))
#     print('No. of dev instances = ', len(dev_batch_loader.dataset))
#     print('No. of test instances = ', len(test_batch_loader.dataset))
#     end = time.time()
#     hours, minutes, seconds = calc_elapsed_time(start, end)
#     print("\n"+ "-"*50 + "\nTook  {:0>2} hours: {:0>2} mins: {:05.2f} secs  to Prepare Data\n".format(hours,minutes,seconds))
#     return train_batch_loader, dev_batch_loader, test_batch_loader, TEXT, LABEL, train_split, val_split, test_split



def prepare_elmo_training(config, fold=None): 
    if fold:
        print("="*100 + "\n\t\t\t\t Preparing Data for fold {}\n".format(fold) + "="*100)
    else:
        print("="*100 + "\n\t\t\t\t\t Preparing Data\n" + "="*100)
        
    start = time.time()
    print("\n\n==>> Loading Data splits and tokenizing each document....")
    
    if config['data_name'] != 'pheme':
        train_data_dir = os.path.join(config['data_path'], 'train.tsv')
        val_data_dir = os.path.join(config['data_path'], 'val.tsv')
        test_data_dir = os.path.join(config['data_path'], 'test.tsv')
    else:
        if fold==0:
            train_data_dir = os.path.join(config['data_path'], 'train_1.tsv')
            val_data_dir = os.path.join(config['data_path'], 'val.tsv')
            test_data_dir = os.path.join(config['data_path'], 'val.tsv')
        else:
            train_data_dir = os.path.join(config['data_path'], 'train_{}.tsv'.format(fold))
            val_data_dir = os.path.join(config['data_path'], 'test_{}.tsv'.format(fold))
            test_data_dir = os.path.join(config['data_path'], 'test_{}.tsv'.format(fold))
       
    directories = ['train_data_dir', 'val_data_dir', 'test_data_dir']
    train_data, val_data, test_data, train_labels, val_labels, test_labels = [], [], [], [], [], []
    
    if config['model_name'] != 'han':
        for i in range(len(directories)):
            data_dir = eval(directories[i])
            df = pandas.read_csv(data_dir, sep='\t', encoding = 'utf8', header=None)
            # Each ele is a document, that we tokenize
            if i==0:
                train_data = [nltk.word_tokenize(ele) for ele in list(df[0])]
                train_labels = [ele for ele in list(df[1])]
            elif i==1:
                val_data = [nltk.word_tokenize(ele) for ele in list(df[0])]
                val_labels = [ele for ele in list(df[1])]
            else:
                test_data = [nltk.word_tokenize(ele) for ele in list(df[0])]
                test_labels = [ele for ele in list(df[1])]            
    print("\n\n" + "-"*50 + "\nDATA STATISTICS:\n" + "-"*50)
    print('No. of target classes = ', config["n_classes"])
    print('No. of train instances = ', len(train_labels))
    print('No. of dev instances = ', len(val_labels))
    print('No. of test instances = ', len(test_labels))
    
    end = time.time()
    hours, minutes, seconds = calc_elapsed_time(start, end)
    print("\n"+ "-"*50 + "\nTook  {:0>2} hours: {:0>2} mins: {:05.2f} secs  to Prepare Data\n".format(hours,minutes,seconds))           
    return train_data, train_labels, val_data, val_labels, test_data, test_labels
    
    
    
    

def prepare_glove_training(config, fold=None):
    if fold:
        print("="*100 + "\n\t\t\t\t Preparing Data for fold {}\n".format(fold) + "="*100)
    else:
        print("="*100 + "\n\t\t\t\t\t Preparing Data\n" + "="*100)
        
    start = time.time()
    if config['data_name'] in ['politifact', 'gossipcop']:
        if config['model_name'] == 'han':
            TEXT, LABEL, train_batch_loader, dev_batch_loader, test_batch_loader, train_split, val_split, test_split = FakeNews_HAN.main_handler(config, config['data_path'], shuffle=True)
        else:
            TEXT, LABEL, train_batch_loader, dev_batch_loader, test_batch_loader, train_split, val_split, test_split = FakeNews.main_handler(config, config['data_path'], shuffle=True)
    
    elif config['data_name'] in ['HealthStory', 'HealthRelease']:
        if config['model_name'] == 'han':
            TEXT, LABEL, train_batch_loader, dev_batch_loader, test_batch_loader, train_split, val_split, test_split = FakeHealth_HAN.main_handler(config, config['data_path'], fold, shuffle=True)
        else:
            TEXT, LABEL, train_batch_loader, dev_batch_loader, test_batch_loader, train_split, val_split, test_split = FakeHealth.main_handler(config, config['data_path'], fold, shuffle=True)
            
    elif config['data_name'] == 'pheme':
        if config['model_name'] == 'han':
            TEXT, LABEL, train_batch_loader, dev_batch_loader, test_batch_loader, train_split, val_split, test_split = Rumor_HAN.main_handler(config, config['data_path'], fold, shuffle=True)
        else:
            TEXT, LABEL, train_batch_loader, dev_batch_loader, test_batch_loader, train_split, val_split, test_split = Rumor.main_handler(config, config['data_path'], fold, shuffle=True)
    
    
    vocab_size = len(TEXT.vocab)
    config['vocab_size'] = vocab_size
    print("\n\n" + "-"*50 + "\nDATA STATISTICS:\n" + "-"*50)
    print("\nVocabulary size = ", vocab_size)
    print('No. of target classes = ', train_batch_loader.dataset.NUM_CLASSES)
    print('No. of train instances = ', len(train_batch_loader.dataset))
    print('No. of dev instances = ', len(dev_batch_loader.dataset))
    print('No. of test instances = ', len(test_batch_loader.dataset))
    end = time.time()
    hours, minutes, seconds = calc_elapsed_time(start, end)
    print("\n"+ "-"*50 + "\nTook  {:0>2} hours: {:0>2} mins: {:05.2f} secs  to Prepare Data\n".format(hours,minutes,seconds))
    return train_batch_loader, dev_batch_loader, test_batch_loader, TEXT, LABEL, train_split, val_split, test_split




def calc_elapsed_time(start, end):
    hours, rem = divmod(end-start, 3600)
    time_hours, time_rem = divmod(end, 3600)
    minutes, seconds = divmod(rem, 60)
    time_mins, _ = divmod(time_rem, 60)
    return int(hours), int(minutes), seconds


def evaluation_measures_pheme(config, preds, labels):
    f1_micro = f1_score(labels, preds, average = 'micro')
    f1_macro = f1_score(labels, preds, average = 'macro')
    f1_weighted = f1_score(labels, preds, average = 'weighted')
    # target_names = ['False', 'True', 'Unverif']
    # print(classification_report(labels, preds, target_names=target_names))
    
    # f1_true = f1_score(labels, preds, average = 'binary', pos_label=1)
    # f1_macro = f1_score(labels, preds, average = 'macro')
    # f1_false = f1_score(labels, preds, average = 'binary', pos_label=0)
    # f1_unverif = f1_score(labels, preds, average = 'binary', pos_label=2)
    
    recall_macro = recall_score(labels, preds, average = 'macro')
    recall_micro = recall_score(labels, preds, average = 'micro')
    recall_weighted = recall_score(labels, preds, average = 'weighted')
    
    precision_micro = precision_score(labels, preds, average = 'micro')
    precision_macro = precision_score(labels, preds, average = 'macro')
    precision_weighted = precision_score(labels, preds, average = 'weighted')
    
    accuracy = accuracy_score(labels, preds)
    return (f1_micro, f1_macro, f1_weighted), (recall_micro, recall_macro, recall_weighted), (precision_micro, precision_macro, precision_weighted), accuracy
    # return (f1_true, f1_macro, f1_false, f1_unverif), (recall_micro, recall_macro, recall_weighted), (precision_micro, precision_macro, precision_weighted), accuracy


def evaluation_measures(config, preds, labels):
    f1 = f1_score(labels, preds, average = 'binary', pos_label =1)
    f1_macro = f1_score(labels, preds, average = 'macro')
    recall = recall_score(labels, preds, average = 'binary', pos_label=1)
    precision = precision_score(labels, preds, average = 'binary', pos_label=1)
    accuracy = accuracy_score(labels, preds)
    # print(metrics.classification_report(labels, preds))
    return f1, f1_macro, recall, precision, accuracy


def evaluation_measures_pheme_filtered(config, preds, labels):
    f1 = f1_score(labels, preds, average = 'binary', pos_label =1)
    f1_neg = f1_score(labels, preds, average ='binary', pos_label=0)
    f1_macro = f1_score(labels, preds, average = 'macro')
    recall = recall_score(labels, preds, average = 'binary', pos_label=1)
    precision = precision_score(labels, preds, average = 'binary', pos_label=1)
    accuracy = accuracy_score(labels, preds)
    # print(metrics.classification_report(labels, preds))
    return f1, f1_neg, f1_macro, recall, precision, accuracy


def log_tensorboard(config, writer, model, epoch, iters, total_iters, loss, f1, prec, recall, acc, lr=0, thresh=0, loss_only=True, val=False):   
    if config['parallel_computing'] == True:
        model_log = model.module
    else:
        model_log = model
        
    if loss_only:
        writer.add_scalar('Train/Loss', sum(loss)/len(loss), ((iters+1)+ total_iters))
        # if iters%500 == 0:
        #     for name, param in model_log.encoder.named_parameters():
        #         print("\nparam {} grad = {}".format(name, param.grad.data.view(-1)))
        #         sys.exit()
        #         if not param.requires_grad or param.grad is None:
        #             continue
        #         writer.add_histogram('Iters/'+name, param.data.view(-1), global_step= ((iters+1)+total_iters))
        #         writer.add_histogram('Grads/'+ name, param.grad.data.view(-1), global_step = ((iters+1)+ total_iters))
    else:
        if not val and config['data_name'] != 'pheme':
            writer.add_scalar('Train/F1', f1, epoch)
            writer.add_scalar('Train/Precision', prec, epoch)
            writer.add_scalar('Train/Recall', recall, epoch)
            writer.add_scalar('Train/Accuracy', acc, epoch)
            writer.add_scalar("Train/learning_rate", lr, epoch)
            
            # for name, param in model_log.encoder.named_parameters():
            #     if not param.requires_grad:
            #         continue
            #     writer.add_histogram('Epochs/' + name, param.data.view(-1), global_step= epoch)
        
        elif not val and config['data_name'] == 'pheme':
            f1_micro, f1_macro, f1_weighted = f1
            recall_micro, recall_macro, recall_weighted = recall
            precision_micro, precision_macro, precision_weighted = prec
            
            writer.add_scalar('Train_F1/macro', f1_macro, epoch)
            writer.add_scalar('Train_F1/micro', f1_micro, epoch)
            writer.add_scalar('Train_F1/weighted', f1_weighted, epoch)
            
            writer.add_scalar('Train_Precision/macro', precision_macro, epoch)
            writer.add_scalar('Train_Precision/micro', precision_micro, epoch)
            writer.add_scalar('Train_Precision/weighted', precision_weighted, epoch)            
            
            writer.add_scalar('Train_Recall/macro', recall_macro, epoch)
            writer.add_scalar('Train_Recall/micro', recall_micro, epoch)
            writer.add_scalar('Train_Recall/weighted', recall_weighted, epoch)
            
            writer.add_scalar("Train/learning_rate", lr, epoch)
            
            # for name, param in model_log.encoder.named_parameters():
            #     if not param.requires_grad:
            #         continue
            #     writer.add_histogram('Epochs/' + name, param.data.view(-1), global_step= epoch)
            
        elif val and config['data_name'] != 'pheme':
            writer.add_scalar('Validation/Loss', loss, epoch)
            writer.add_scalar('Validation/F1', f1, epoch)
            writer.add_scalar('Validation/Recall', recall, epoch)
            writer.add_scalar('Validation/Precision', prec, epoch)
            writer.add_scalar('Validation/Accuracy', acc, epoch)
        
        elif val and config['data_name'] == 'pheme':
            f1_micro, f1_macro, f1_weighted = f1
            recall_micro, recall_macro, recall_weighted = recall
            precision_micro, precision_macro, precision_weighted = prec
            
            writer.add_scalar('Validation/Loss', loss, epoch)
            
            writer.add_scalar('Validation_F1/macro', f1_macro, epoch)
            writer.add_scalar('Validation_F1/micro', f1_micro, epoch)
            writer.add_scalar('Validation_F1/weighted', f1_weighted, epoch)
            
            writer.add_scalar('Validation_Precision/macro', precision_macro, epoch)
            writer.add_scalar('Validation_Precision/micro', precision_micro, epoch)
            writer.add_scalar('Validation_Precision/weighted', precision_weighted, epoch)            
            
            writer.add_scalar('Validation_Recall/macro', recall_macro, epoch)
            writer.add_scalar('Validation_Recall/micro', recall_micro, epoch)
            writer.add_scalar('Validation_Recall/weighted', recall_weighted, epoch)
            
                

def print_stats(config, epoch, train_loss, train_acc, train_f1, train_f1_macro, train_prec, train_recall, val_loss, val_acc, val_f1, val_f1_macro, val_precision, val_recall, start, lr):
    end = time.time()
    hours, minutes, seconds = calc_elapsed_time(start, end)
    
    train_loss = sum(train_loss)/len(train_loss)
    print("\nEpoch: {}/{},  \
          \ntrain_loss = {:.4f},    train_acc = {:.4f},    train_prec = {:.4f},    train_recall = {:.4f},    train_f1 = {:.4f},    train_macro_f1 = {:.4f}  \
          \neval_loss = {:.4f},     eval_acc = {:.4f},     eval_prec = {:.4f},     eval_recall = {:.4f},     eval_f1 = {:.4f},    val_f1_macro = {:.4f}  \
              \nlr  =  {:.8f}\nElapsed Time:  {:0>2}:{:0>2}:{:05.2f}"
                     .format(epoch, config['max_epoch'], train_loss, train_acc, train_prec, train_recall, train_f1, train_f1_macro, val_loss, val_acc, 
                             val_precision, val_recall, val_f1, val_f1_macro, lr, hours,minutes,seconds))
        
        

def print_stats_pheme_filtered(config, epoch, train_loss, train_acc, train_f1, train_f1_macro, train_prec, train_recall, val_loss, val_acc, val_f1, val_f1_macro, val_precision, val_recall, start, lr):
    end = time.time()
    hours, minutes, seconds = calc_elapsed_time(start, end)
    
    train_loss = sum(train_loss)/len(train_loss)
    print("\nEpoch: {}/{},  \
          \ntrain_loss = {:.4f},    train_acc = {:.4f},    train_prec = {:.4f},    train_recall = {:.4f},    train_f1 = {:.4f},    train_macro_f1 = {:.4f}  \
          \neval_loss = {:.4f},     eval_acc = {:.4f},     eval_prec = {:.4f},     eval_recall = {:.4f},     eval_f1 = {:.4f},    val_f1_macro = {:.4f}  \
              \nlr  =  {:.8f}\nElapsed Time:  {:0>2}:{:0>2}:{:05.2f}"
                     .format(epoch, config['max_epoch'], train_loss, train_acc, train_prec, train_recall, train_f1, train_f1_macro, val_loss, val_acc, 
                             val_precision, val_recall, val_f1, val_f1_macro, lr, hours,minutes,seconds))
        
        

def print_stats_pheme(config, epoch, train_loss, train_acc, train_f1, train_prec, train_recall, val_loss, val_acc, val_f1, val_prec, val_recall, start, lr):
    end = time.time()
    hours, minutes, seconds = calc_elapsed_time(start, end)
    
    train_f1_micro, train_f1_macro, train_f1_weighted = train_f1
    train_recall_micro, train_recall_macro, train_recall_weighted = train_recall
    train_precision_micro, train_precision_macro, train_precision_weighted = train_prec
    
    val_f1_micro, val_f1_macro, val_f1_weighted = val_f1
    val_recall_micro, val_recall_macro, val_recall_weighted = val_recall
    val_precision_micro, val_precision_macro, val_precision_weighted = val_prec
    
    if config['training_setting'] == 'mtl':
        task = 'Veracity' if (epoch-1)%2==0 else 'Stance'
    else:
        task= "Veracity"
    train_loss = sum(train_loss)/len(train_loss)
    print("\nEpoch: {}/{},  \nTASK = {}\
          \ntrain_loss = {:.4f}, \
          \ntrain_micro_prec = {:.4f},      train_macro_prec = {:.4f},      train_weighted_prec = {:.4f}\
          \ntrain_micro_recall = {:.4f},    train_macro_recall = {:.4f},    train_weighted_recall = {:.4f}\
          \ntrain_micro_f1 = {:.4f},        train_macro_f1 = {:.4f},        train_weighted_f1 = {:.4f}\
          \neval_loss = {:.4f},\
          \nval_micro_prec = {:.4f},      val_macro_prec = {:.4f},      val_weighted_prec = {:.4f}\
          \nval_micro_recall = {:.4f},    val_macro_recall = {:.4f},    val_weighted_recall = {:.4f}\
          \nval_micro_f1 = {:.4f},        val_macro_f1 = {:.4f},        val_weighted_f1 = {:.4f}\
          \nlr  =  {:.8f} \
          \nElapsed Time:  {:0>2}:{:0>2}:{:05.2f}"
                     .format(epoch, config['max_epoch'], task, train_loss, train_precision_micro, train_precision_macro, train_precision_weighted, train_recall_micro, train_recall_macro, train_recall_weighted,
                             train_f1_micro, train_f1_macro, train_f1_weighted, val_loss, val_precision_micro, val_precision_macro, val_precision_weighted,
                             val_recall_micro, val_recall_macro, val_recall_weighted, val_f1_micro, val_f1_macro, val_f1_weighted, lr, hours,minutes,seconds))


def print_test_stats(test_accuracy, test_precision, test_recall, test_f1, test_f1_macro, best_val_acc, best_val_precision, best_val_recall, best_val_f1):
    print("\nTest accuracy of best model = {:.2f}".format(test_accuracy*100))
    print("Test precision of best model = {:.2f}".format(test_precision*100))
    print("Test recall of best model = {:.2f}".format(test_recall*100))
    print("Test f1 of best model = {:.2f}".format(test_f1*100))
    print("Test macro-F1 of best model = {:.2f}".format(test_f1_macro*100))
    print("\n" + "-"*50 + "\nBest Validation scores:\n" + "-"*50)
    print("\nVal accuracy of best model = {:.2f}".format(best_val_acc*100))
    print("Val precision of best model = {:.2f}".format(best_val_precision*100))
    print("Val recall of best model = {:.2f}".format(best_val_recall*100))
    print("Val f1 of best model = {:.2f}".format(best_val_f1*100))
    
    
def print_test_stats_pheme(test_accuracy, test_precision, test_recall, test_f1, best_val_acc, best_val_precision, best_val_recall, best_val_f1):
    print("\nTest accuracy of best model = {:.2f}".format(test_accuracy*100))
    print("Test (macro) precision of best model = {:.2f}".format(test_precision[1]*100))
    print("Test (macro) recall of best model = {:.2f}".format(test_recall[1]*100))
    print("Test (micro) f1 of best model = {:.2f}".format(test_f1[0]*100))
    print("Test (macro) f1 of best model = {:.2f}".format(test_f1[1]*100))
    print("Test (weighted) f1 of best model = {:.2f}".format(test_f1[2]*100))
    print("\n" + "-"*50 + "\nBest Validation scores:\n" + "-"*50)
    print("\nVal accuracy of best model = {:.2f}".format(best_val_acc*100))
    print("Val (macro) precision of best model = {:.2f}".format(best_val_precision[1]*100))
    print("Val (macro) recall of best model = {:.2f}".format(best_val_recall[1]*100))
    print("Val (micro) f1 of best model = {:.2f}".format(best_val_f1[0]*100))
    print("Val (macro) f1 of best model = {:.2f}".format(best_val_f1[1]*100))
    print("Val (weighted) f1 of best model = {:.2f}".format(best_val_f1[2]*100))
    
    
def calculate_transformer_stats(train_result):
    train_prec_pos = train_result['tp']/ (train_result['tp'] + train_result['fp'])
    train_recall_pos = train_result['tp']/ (train_result['tp'] + train_result['fn'])
    train_f1_pos = (2*train_prec_pos*train_recall_pos)/ (train_prec_pos + train_recall_pos)               
    train_prec_neg = train_result['tn']/ (train_result['tn'] + train_result['fn'])
    train_recall_neg = train_result['tn']/ (train_result['tn'] + train_result['fp'])
    train_f1_neg = (2*train_prec_neg*train_recall_neg)/ (train_prec_neg + train_recall_neg)
    macro_f1 = (train_f1_pos + train_f1_neg) / 2
    return train_prec_pos, train_recall_pos, train_f1_pos, macro_f1


def print_transformer_results(config, val_stats, test_stats, val_result, test_result):
    
    if config['data_name'] != 'pheme':
        val_f1_pos, val_f1_neg, val_macro_f1, val_micro_f1, val_recall, val_prec, val_acc = val_stats
        test_f1_pos, test_f1_neg, test_macro_f1, test_micro_f1, test_recall, test_prec, test_acc = test_stats
        val_mcc = val_result['mcc']
        test_mcc = test_result['mcc']   
        
        print("\nVal evaluation stats: \n" + "-"*50)
        print("Val precision of best model = {:.2f}".format(val_prec*100))
        print("Val recall of best model = {:.2f}".format(val_recall*100))
        print("Val f1 (fake) of best model = {:.2f}".format(val_f1_pos*100))
        print("Val f1 (real) of best model = {:.2f}".format(val_f1_neg*100))
        print("Val macro-f1 of best model = {:.2f}".format(val_macro_f1*100))
        print("Val micro-f1 of best model = {:.2f}".format(val_micro_f1*100))
        print("Val accuracy of best model = {:.2f}".format(val_acc*100))
        print("Val MCC of best model = {:.2f}".format(val_mcc*100))
        
        print("\nTest evaluation stats: \n" + "-"*50)
        print("Test precision of best model = {:.2f}".format(test_prec*100))
        print("Test recall of best model = {:.2f}".format(test_recall*100))
        print("Test f1 (fake) of best model = {:.2f}".format(test_f1_pos*100))
        print("Test f1 (real) of best model = {:.2f}".format(test_f1_neg*100))
        print("Test macro-f1 of best model = {:.2f}".format(test_macro_f1*100))
        print("Test micro-f1 of best model = {:.2f}".format(test_micro_f1*100))
        print("Test accuracy of best model = {:.2f}".format(test_acc*100))
        print("Test MCC of best model = {:.2f}".format(test_mcc*100))
        
    else:
        val_f1, val_recall, val_prec, val_accuracy = val_stats
        val_f1_micro, val_f1_macro, val_f1_weighted = val_f1
        val_recall_micro, val_recall_macro, val_recall_weighted = val_recall
        val_precision_micro, val_precision_macro, val_precision_weighted = val_prec
        
        test_f1, test_recall, test_prec, test_accuracy = test_stats
        test_f1_micro, test_f1_macro, test_f1_weighted = test_f1
        test_recall_micro, test_recall_macro, test_recall_weighted = test_recall
        test_precision_micro, test_precision_macro, test_precision_weighted = test_prec
        
        print_test_stats_pheme(test_accuracy, test_prec, test_recall, test_f1, val_accuracy, val_prec, val_recall, val_f1)
        


def print_transformer_results_pheme(config, val_stats, test_stats, val_result, test_result):
    
    val_f1_pos, val_f1_neg, val_macro_f1, val_micro_f1, val_recall, val_prec, val_acc = val_stats
    test_f1_pos, test_f1_neg, test_macro_f1, test_micro_f1, test_recall, test_prec, test_acc = test_stats
    val_mcc = val_result['mcc']
    test_mcc = test_result['mcc']   
    
    print("\nVal evaluation stats: \n" + "-"*50)
    print("Val precision of best model = {:.2f}".format(val_prec*100))
    print("Val recall of best model = {:.2f}".format(val_recall*100))
    print("Val f1 (fake) of best model = {:.2f}".format(val_f1_pos*100))
    print("Val f1 (real) of best model = {:.2f}".format(val_f1_neg*100))
    print("Val macro-f1 of best model = {:.2f}".format(val_macro_f1*100))
    print("Val micro-f1 of best model = {:.2f}".format(val_micro_f1*100))
    print("Val accuracy of best model = {:.2f}".format(val_acc*100))
    print("Val MCC of best model = {:.2f}".format(val_mcc*100))
    
    print("\nTest evaluation stats: \n" + "-"*50)
    print("Test precision of best model = {:.2f}".format(test_prec*100))
    print("Test recall of best model = {:.2f}".format(test_recall*100))
    print("Test f1 (fake) of best model = {:.2f}".format(test_f1_pos*100))
    print("Test f1 (real) of best model = {:.2f}".format(test_f1_neg*100))
    print("Test macro-f1 of best model = {:.2f}".format(test_macro_f1*100))
    print("Test micro-f1 of best model = {:.2f}".format(test_micro_f1*100))
    print("Test accuracy of best model = {:.2f}".format(test_acc*100))
    print("Test MCC of best model = {:.2f}".format(test_mcc*100))
        