import os, time, json, sys, gc
import numpy as np
import nltk, pandas
import torch
from scipy.sparse import csr_matrix, lil_matrix, save_npz, load_npz
from torch_geometric.data import Data, DataLoader, GraphSAINTRandomWalkSampler, GraphSAINTNodeSampler, GraphSAINTEdgeSampler
from torch_geometric.data.cluster import ClusterData, ClusterLoader
from torch_geometric.utils import remove_isolated_nodes, contains_isolated_nodes, contains_self_loops

import warnings
warnings.filterwarnings("ignore")
sys.path.append("..")
from utils.data_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



##############################################
#            Main GNN-DATA Hadler            #
##############################################

class Prepare_GNN_Dataset():
    def __init__(self, config):
        super(Prepare_GNN_Dataset, self).__init__()
        self.config = config
        self.read_files()
        

    def read_files(self, verbose=True):
        start = time.time()
        if verbose:
            print("="*100 + "\n\t\t\t\t Preparing Data for {}\n".format(self.config['data_name']) + "="*100)
            print("\n\n==>> Loading feature matrix and adj matrix....") 
            
        if self.config['data_name'] in ['gossipcop', 'politifact']:
            x_file = os.path.join(self.config['data_path'], self.config['data_name'], 'feat_matrix_lr_train_30_5.npz'.format(self.config['data_name']))
            y_file = os.path.join(self.config['data_path'], self.config['data_name'], 'all_labels_lr_train_30_5.json'.format(self.config['data_name']))
            # adj_name = 'adj_matrix_lr_train_30_5_edge.npy'.format(self.config['data_name']) if self.config['model_name'] != 'HGCN' else 'adj_matrix_lr_train_30_5.npz'.format(self.config['data_name'])
            adj_name = 'adj_matrix_lr_train_30_5_edge.npy'.format(self.config['data_name'])
            edge_index_file = os.path.join(self.config['data_path'], self.config['data_name'], adj_name)
            node2id_file = os.path.join(self.config['data_path'], self.config['data_name'], 'node2id_lr_train_30_5.json'.format(self.config['data_name']))
            node_type_file = os.path.join(self.config['data_path'], self.config['data_name'], 'node_type_lr_train_30_5.npy'.format(self.config['data_name']))
            split_mask_file = os.path.join(self.config['data_path'], self.config['data_name'], 'split_mask_lr_30_5.json')
            if self.config['model_name'] in ['rgcn', 'rgat', 'rsage']:
                edge_type_file = os.path.join(self.config['data_path'], self.config['data_name'], 'edge_type_lr_train_30_5_edge.npy'.format(self.config['data_name']))
        else:
            x_file = os.path.join(self.config['data_path'], self.config['data_name'], 'feat_matrix_lr_top10_train.npz')
            y_file = os.path.join(self.config['data_path'], self.config['data_name'], 'all_labels_lr_top10_train.json')
            # adj_name = 'adj_matrix_lr_top10_train_edge.npy' if self.config['model_name'] != 'HGCN' else 'adj_matrix_lr_top10_train.npz'
            adj_name = 'adj_matrix_lr_top10_train_edge.npy'
            edge_index_file = os.path.join(self.config['data_path'], self.config['data_name'], adj_name)
            node2id_file = os.path.join(self.config['data_path'], self.config['data_name'], 'node2id_lr_top10_train.json')
            node_type_file = os.path.join(self.config['data_path'], self.config['data_name'], 'node_type_lr_top10_train.npy')
            split_mask_file = os.path.join(self.config['data_path'], self.config['data_name'], 'split_mask_top10.json')
            if self.config['model_name'] in ['rgcn', 'rgat', 'rsage']:
                edge_type_file = os.path.join(self.config['data_path'], self.config['data_name'], 'edge_type_lr_top10_edge.npy')
        
    
        # if self.config['model_name'] != 'HGCN':
        #     edge_index_data = np.load(edge_index_file)
        #     edge_index_data = torch.from_numpy(edge_index_data).long()
            
        # elif self.config['model_name'] == 'HGCN':
        #     edge_index_data = load_npz(edge_index_file)
            
        #     # edge_index_data = torch.from_numpy(edge_index_data.toarray())
            
        #     edge_index_data = edge_index_data.tocoo()
        #     indices = torch.from_numpy(np.vstack((edge_index_data.row, edge_index_data.col)).astype(np.int64))
        #     values = torch.Tensor(edge_index_data.data)
        #     shape = torch.Size(edge_index_data.shape)
        #     edge_index_data = torch.sparse.FloatTensor(indices, values, shape)
        
        self.edge_index_data = np.load(edge_index_file)
        self.edge_index_data = torch.from_numpy(edge_index_data).long()
        
        self.x_data = load_npz(x_file)
        self.x_data = torch.from_numpy(self.x_data.toarray())
        self.y_data = json.load(open(y_file, 'r'))
        self.y_data = torch.LongTensor(self.y_data['all_labels'])
        self.node2id = json.load(open(node2id_file, 'r'))
        # node_type = np.load(node_type_file)
        # node_type = torch.from_numpy(node_type).float()
        if self.config['model_name'] in ['rgcn', 'rgat', 'rsage']:
            self.edge_type_data = np.load(edge_type_file)
            self.edge_type_data = torch.from_numpy(self.edge_type_data).long()
        else:
            self.edge_type_data = None
        
        self.split_masks = json.load(open(split_mask_file, 'r'))
        
        num_nodes, self.vocab_size = self.x_data.shape
        if self.config['model_name'] != 'HGCN':
            isolated_nodes = contains_isolated_nodes(edge_index= self.edge_index_data)
            self_loops = contains_self_loops(edge_index= self.edge_index_data)
        
        if verbose:
            print("\n\n" + "-"*50 + "\nDATA STATISTICS:\n" + "-"*50)
            if self.config['model_name'] != 'HGCN':
                print("Contains isolated nodes = ", isolated_nodes)
                print("Contains self loops = ", self_loops)
            print("Vocabulary size = ", self.vocab_size)
            print('No. of nodes in graph = ', num_nodes)
            print('No. of nodes after removing isolated nodes = ', new_num_nodes)
            print("No. of edges in graph = ", self.data.num_edges)
            
            print("\nNo.of train instances = ", self.data.train_mask.sum().item())
            print("No.of val instances = ", self.data.val_mask.sum().item())
            print("No.of test instances = ", num_nodes - self.data.train_mask.sum().item() - self.data.val_mask.sum().item())
                
            end = time.time()
            hours, minutes, seconds = calc_elapsed_time(start, end)
            print("\n"+ "-"*50 + "\nTook  {:0>2} hours: {:0>2} mins: {:05.2f} secs  to Prepare Data\n".format(hours,minutes,seconds))
    

    
    
    
    def prepare_gnn_training(self):
        if verbose:
            print("\n\n==>> Clustering the graph and preparing dataloader....")
            
        self.data = Data(x=self.x_data.float(), edge_index = self.edge_index_data.long(), edge_attr = self.edge_type_data, y=self.y_data)
        new_num_nodes, _ = self.data.x.shape
        
        self.data.train_mask = torch.FloatTensor(self.split_masks['train_mask'])
        self.data.val_mask = torch.FloatTensor(self.split_masks['val_mask'])
        self.data.representation_mask = torch.FloatTensor(self.split_masks['repr_mask']) 
        self.data.node2id = torch.tensor(list(self.node2id.values()))
        # self.data.node_type = self.node_type
            
        
        if not self.config['full_graph']:
            if self.config['cluster'] :
                cluster_data = ClusterData(self.data, num_parts=self.config['clusters'], recursive=False)
                self.loader = ClusterLoader(cluster_data, batch_size=self.config['batch_size'], shuffle=self.config['shuffle'], num_workers=0)
            elif self.config['saint'] == 'random_walk':
                self.loader = GraphSAINTRandomWalkSampler(self.data, batch_size=6000, walk_length=2, num_steps=5, sample_coverage=100, num_workers=0)
            elif self.config['saint'] == 'node':
                self.loader = GraphSAINTNodeSampler(self.data, batch_size=6000, num_steps=5, sample_coverage=100, num_workers=0)
            elif self.config['saint'] == 'edge':
                self.loader = GraphSAINTEdgeSampler(self.data, batch_size=6000, num_steps=5, sample_coverage=100, num_workers=0)
        else:
            self.loader=None
        

        return self.loader, self.vocab_size, self.data