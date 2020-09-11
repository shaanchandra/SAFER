from __future__ import absolute_import, division, print_function

import torch, math, os, sys
from torch import nn

import warnings
warnings.filterwarnings("ignore")

import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GraphConv, GATConv, RGCNConv
sys.path.append("..")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




#####################################
##  Class for GNN implementations  ##
#####################################
    
    
    
class Graph_Net(torch.nn.Module):
    def __init__(self, config):
        super(Graph_Net, self).__init__()
        self.in_dim = config['vocab_size']
        self.embed_dim = config['embed_dim']
        self.dropout = config['dropout']
        self.fc_dim = config['fc_dim']
        self.node_drop = config['node_drop']
        self.model = config['model_name']
        
        if config['model_name'] == 'graph_sage':
            self.conv1 = SAGEConv(self.in_dim, self.embed_dim, normalize=True)
            self.conv2 = SAGEConv(self.embed_dim, self.embed_dim, normalize=True)
            # self.conv3 = SAGEConv(self.embed_dim, config['n_classes'], normalize=False)
            
        elif config['model_name'] == 'gcn':
            self.conv1 = GCNConv(self.in_dim, self.embed_dim, improved=True)
            self.conv2 = GCNConv(self.embed_dim, self.embed_dim, improved=True)
            # self.conv2 = GCNConv(self.embed_dim, config['n_classes'], improved=True)
        
        elif config['model_name'] == 'graph_conv':
            self.conv1 = GraphConv(self.in_dim, self.embed_dim, aggr='add')
            self.conv2 = GraphConv(self.embed_dim, self.embed_dim, aggr='add')
            # self.conv2 = GraphConv(self.embed_dim, config['n_classes'], aggr='add')
        
        elif config['model_name'] == 'gat':
            self.conv1 = GATConv(self.in_dim, self.embed_dim, heads=3, concat=True, dropout=0.1)
            self.conv2 = GATConv(3*self.embed_dim, self.embed_dim, heads=3, concat=True, dropout=0.1)
            # self.conv2 = GATConv(3*self.embed_dim, config['n_classes'], heads=3, concat=False, dropout=0.1)
        
        if config['model_name'] == 'gat':
            self.classifier = nn.Sequential(nn.Dropout(config["dropout"]), 
                                                nn.Linear(3*self.embed_dim, self.fc_dim), 
                                                nn.ReLU(), 
                                                nn.Linear(self.fc_dim, config['n_classes']))
        else:
             self.classifier = nn.Sequential(nn.Dropout(config["dropout"]), 
                                            nn.Linear(self.embed_dim, self.fc_dim), 
                                            nn.ReLU(), 
                                            nn.Linear(self.fc_dim, config['n_classes']))   
            

    def forward(self, x, edge_index, edge_type = None):
        
        node_mask = torch.FloatTensor(x.shape[0], 1).uniform_() > self.node_drop
        if self.training:
            # noise1 = torch.FloatTensor(x.shape[0], x.shape[1]).uniform_() > 0.7
            # noise2 = torch.FloatTensor(x.shape[0], x.shape[1]).uniform_() > 0.3
            # noise2 = 2*torch.rand((x.shape[0], x.shape[1]))
            # noise_mask = torch.FloatTensor(x.shape[0], x.shape[1]).uniform_() > 0.5
            # noise2 = noise2 * noise_mask
            # # x = noise1 * x # Make some zeros
            # x = noise2 + x # Make some ones
            x = node_mask.to(device) * x  #/ (1 - self.node_drop)
        x = F.relu(self.conv1(x.float(), edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x.float(), edge_index)
        out = self.classifier(x)
        return out, node_mask, x
    
    
    
    


class Graph_Net_MTL(torch.nn.Module):
    def __init__(self, config):
        super(Graph_Net_MTL, self).__init__()
        self.in_dim = config['vocab_size']
        self.embed_dim = config['embed_dim']
        self.dropout = config['dropout']
        self.fc_dim = config['fc_dim']
        self.node_drop = config['node_drop']
        self.model = config['model_name']
        self.training_mode = config['mode']
        self.training_setting = config['training_setting']
        
        if config['model_name'] == 'graph_sage':
            self.conv1 = SAGEConv(self.in_dim, self.embed_dim, normalize=True)
            self.conv2 = SAGEConv(self.embed_dim, self.embed_dim, normalize=True)
            # self.conv3 = SAGEConv(self.embed_dim, config['n_classes'], normalize=False)
            
        elif config['model_name'] == 'gcn':
            self.conv1 = GCNConv(self.in_dim, self.embed_dim, improved=True)
            self.conv2 = GCNConv(self.embed_dim, self.embed_dim, improved=True)
            # self.conv2 = GCNConv(self.embed_dim, config['n_classes'], improved=True)
        
        elif config['model_name'] == 'graph_conv':
            self.conv1 = GraphConv(self.in_dim, self.embed_dim, aggr='add')
            self.conv2 = GraphConv(self.embed_dim, self.embed_dim, aggr='add')
            # self.conv2 = GraphConv(self.embed_dim, config['n_classes'], aggr='add')
        
        elif config['model_name'] == 'gat':
            self.conv1 = GATConv(self.in_dim, self.embed_dim, heads=3, concat=True, dropout=0.1)
            self.conv2 = GATConv(3*self.embed_dim, self.embed_dim, heads=3, concat=True, dropout=0.1)
            # self.conv2 = GATConv(3*self.embed_dim, config['n_classes'], heads=3, concat=False, dropout=0.1)
        
        if self.training_setting == 'normal':
            if config['model_name'] == 'gat':
                self.classifier = nn.Sequential(nn.Dropout(config["dropout"]), 
                                                    nn.Linear(3*self.embed_dim, self.fc_dim), 
                                                    nn.ReLU(), 
                                                    nn.Linear(self.fc_dim, config['n_classes']))
            else:
                 self.classifier = nn.Sequential(nn.Dropout(config["dropout"]), 
                                                nn.Linear(self.embed_dim, self.fc_dim), 
                                                nn.ReLU(), 
                                                nn.Linear(self.fc_dim, config['n_classes']))   
        
        
        else:
            if config['model_name'] == 'gat':
                self.classifier_stance = nn.Sequential(nn.Dropout(config["dropout"]), 
                                                    nn.Linear(3*self.embed_dim, self.fc_dim), 
                                                    nn.ReLU(), 
                                                    nn.Linear(self.fc_dim, config['n_classes_stance']))
                self.classifier_veracity = nn.Sequential(nn.Dropout(config["dropout"]), 
                                                    nn.Linear(3*self.embed_dim, self.fc_dim), 
                                                    nn.ReLU(), 
                                                    nn.Linear(self.fc_dim, config['n_classes_veracity']))
            else:
                 self.classifier_stance = nn.Sequential(nn.Dropout(config["dropout"]), 
                                                    nn.Linear(self.embed_dim, self.fc_dim), 
                                                    nn.ReLU(), 
                                                    nn.Linear(self.fc_dim, config['n_classes_stance']))
                 self.classifier_veracity = nn.Sequential(nn.Dropout(config["dropout"]), 
                                                    nn.Linear(self.embed_dim, self.fc_dim), 
                                                    nn.ReLU(), 
                                                    nn.Linear(self.fc_dim, config['n_classes_veracity']))   
            

    def forward(self, x, edge_index, edge_type = None, epoch=0):
        # mode = 'veracity' if epoch%2==0 else 'stance'
        mode = 'stance'
        node_mask = torch.FloatTensor(x.shape[0], 1).uniform_() > self.node_drop
        if self.training:
            # noise1 = torch.FloatTensor(x.shape[0], x.shape[1]).uniform_() > 0.7
            # noise2 = torch.FloatTensor(x.shape[0], x.shape[1]).uniform_() > 0.3
            # noise2 = 2*torch.rand((x.shape[0], x.shape[1]))
            # noise_mask = torch.FloatTensor(x.shape[0], x.shape[1]).uniform_() > 0.5
            # noise2 = noise2 * noise_mask
            # # x = noise1 * x # Make some zeros
            # x = noise2 + x # Make some ones
            x = node_mask.to(device) * x  #/ (1 - self.node_drop)
        x = F.relu(self.conv1(x.float(), edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x.float(), edge_index)
        if mode == 'veracity':
            out = self.classifier_veracity(x)
        else:
            out = self.classifier_stance(x)
        return out, node_mask, x
        




################################################
##  Class for Relational-GNN implementations  ##
################################################
        
    
       
class Relational_GNN(torch.nn.Module):
    def __init__(self, config):
        super(Relational_GNN, self).__init__()
        self.num_rels = config['num_rels']
        self.in_dim = config['vocab_size']
        self.embed_dim = config['embed_dim']
        self.dropout = config['dropout']
        self.fc_dim = config['fc_dim']
        self.node_drop = config['node_drop']
        
        if config['model_name'] == 'rsage':
            self.conv1 = torch.nn.ModuleList([SAGEConv(self.in_dim, self.embed_dim, normalize=True, concat=True) for _ in range(self.num_rels)])
            self.conv2 = torch.nn.ModuleList([SAGEConv(self.embed_dim, self.embed_dim, normalize=True, concat=True) for _ in range(self.num_rels)])
            
        elif config['model_name'] == 'rgcn':
            self.conv1 = torch.nn.ModuleList([GCNConv(self.in_dim, self.embed_dim, improved=True) for _ in range(self.num_rels)])
            self.conv2 = torch.nn.ModuleList([GCNConv(self.embed_dim, self.embed_dim, improved=True) for _ in range(self.num_rels)])
        
        elif config['model_name'] == 'rgat':
            self.conv1 = torch.nn.ModuleList([GATConv(self.in_dim, self.embed_dim, heads=3, concat=True, dropout=0.1) for _ in range(self.num_rels)])
            self.conv1 = torch.nn.ModuleList([GATConv(3*self.embed_dim, self.embed_dim, heads=3, concat=True, dropout=0.1) for _ in range(self.num_rels)])
        
        
        if config['model_name'] == 'rgat':
            self.classifier = nn.Sequential(nn.Dropout(config["dropout"]), 
                                                nn.Linear(3*self.embed_dim, self.fc_dim), 
                                                nn.ReLU(), 
                                                nn.Linear(self.fc_dim, config['n_classes']))
        else:
              self.classifier = nn.Sequential(nn.Dropout(config["dropout"]), 
                                            nn.Linear(self.embed_dim, self.fc_dim), 
                                            nn.ReLU(), 
                                            nn.Linear(self.fc_dim, config['n_classes']))

    def forward(self, x, edge_index, edge_type=None):
        node_mask = torch.FloatTensor(x.shape[0], 1).uniform_() > self.node_drop
        if self.training:
            x = node_mask.to(device) * x  #/ (1 - self.node_drop)
        
        out1=0
        for rels in range(self.num_rels):
            rel_mask = edge_type == rels+1
            out1 += self.conv1[rels](x.float(), edge_index[: , rel_mask])
        
        out2=0
        out1 = F.relu(out1)
        out1 = F.dropout(out1, p=self.dropout, training=self.training)
        for rels in range(self.num_rels):
            rel_mask = edge_type == rels+1
            out2 += self.conv2[rels](out1.float(), edge_index[: , rel_mask])
            
        out = self.classifier(out2)
        return out, node_mask, out2
    
    
    

