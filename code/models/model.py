from __future__ import absolute_import, division, print_function

import torch, math, os, sys
import numpy as np
from torch.nn import Parameter
from torchnlp.nn import WeightDrop
from functools import wraps
from copy import deepcopy
from torch import nn
import torch.utils.data as data
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from allennlp.modules import Elmo
from allennlp.modules.elmo import batch_to_ids
# from simpletransformers.classification import ClassificationModel
from simpletransformers.experimental.classification import classification_model

import warnings
warnings.filterwarnings("ignore")
sys.path.append("..")

import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from torch_geometric.nn import SAGEConv, GCNConv, GraphConv, GATConv, RGCNConv

from utils.utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


######################
## Helper Functions ##
######################

"""
The following 2 implementations are taken from the implementation of LSTM-reg in the HEDWIG framework
(https://github.com/castorini/hedwig/tree/master/models/reg_lstm)

"""

def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
      mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
      masked_embed_weight = mask * embed.weight
    else:
      masked_embed_weight = embed.weight
    if scale:
      masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight
      
    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1
      
    X = torch.nn.functional.embedding(words, masked_embed_weight, padding_idx, embed.max_norm, embed.norm_type, embed.scale_grad_by_freq, embed.sparse)
    return X


class WeightDrop_manual(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False):
        super().__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def null_function(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.null_function

        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.variational:
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda:
                    mask = mask.cuda()
                    mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                    w = torch.nn.Parameter(mask.expand_as(raw_w) * raw_w)
            else:
                w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training).to(device)
            setattr(self.module, name_w, w)

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.variational:
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda:
                    mask = mask.cuda()
                    mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                    w = torch.nn.Parameter(mask.expand_as(raw_w) * raw_w)
            else:
                w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training).to(device)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)




###################
## Model Classes ##
###################

"""
Main class that controls training and calling of other classes based on corresponding model_name

"""
class Document_Classifier(nn.Module):
    def __init__(self, config, pre_trained_embeds = None):
        super(Document_Classifier, self).__init__()

        self.lstm_dim = config['lstm_dim']
        self.model_name = config['model_name']
        self.fc_dim = config['fc_dim']
        self.num_classes = config['n_classes']
        self.embed_dim = config['embed_dim'] if config['embed_dim']==300 else 2*config['embed_dim']
        self.batch_size = config['batch_size']
        self.num_kernels = config["kernel_num"]
        self.kernel_sizes = [int(k) for k in config["kernel_sizes"].split(',')]
        self.mode = 'single' if not config['parallel_computing'] else 'multi'
        
        # Choose the right embedding method based on embed_dim given
        if config['embed_dim'] ==300:
            self.vocab_size = config['vocab_size']
            self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
            self.embedding.weight.data.copy_(pre_trained_embeds)
            self.embedding.requires_grad = False
        elif config['embed_dim'] == 128:
            # Small
            self.options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
            self.weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
            self.elmo = Elmo(options_file = self.options_file, weight_file = self.weight_file, num_output_representations=1, requires_grad = False)
        elif config['embed_dim'] == 256:
            # Medium
            self.options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json"
            self.weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"
            self.elmo = Elmo(options_file = self.options_file, weight_file = self.weight_file, num_output_representations=1, requires_grad = False)
        elif config['embed_dim'] == 512:
            # Highest
            self.options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
            self.weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
            self.elmo = Elmo(options_file = self.options_file, weight_file = self.weight_file, num_output_representations=1, requires_grad = False)
        
        
        if self.model_name == 'bilstm':
            self.encoder = BiLSTM(config)
            self.fc_inp_dim = 2*config['lstm_dim']
        elif self.model_name == 'bilstm_pool':
            self.encoder = BiLSTM(config, max_pool = True)
            self.fc_inp_dim = 2*config['lstm_dim']
        elif self.model_name == 'bilstm_reg':
            self.encoder = BiLSTM_reg(config)
            self.fc_inp_dim = config['lstm_dim']
        elif self.model_name == 'han':
            self.encoder = HAN(config)
            self.fc_inp_dim = 2*config['sent_lstm_dim']
        elif self.model_name == 'cnn':
            self.encoder = Kim_CNN(config)
            self.fc_inp_dim = self.num_kernels * len(self.kernel_sizes)

        self.classifier = nn.Sequential(nn.Dropout(config["dropout"]), 
                                            nn.Linear(self.fc_inp_dim, self.fc_dim), 
                                            nn.ReLU(), 
                                            nn.Linear(self.fc_dim, self.num_classes))
        

    def forward(self, inp, sent_lens, doc_lens = 0, arg=0, cache=False):

        if self.model_name in ['bilstm' , 'bilstm_pool', 'cnn']:
            if self.embed_dim ==300:
                inp = self.embedding(inp)
            else:
                inp = self.elmo(inp.contiguous())['elmo_representations'][0]
 
            out = self.encoder(inp.contiguous(), lengths=sent_lens)
            if not cache:
                out = self.classifier(out)
        # for HAN the embeddings are taken care of in its model class
        else:
            if self.embed_dim ==300:
                inp = self.embedding(inp)
                # out = self.encoder(inp, embedding = self.embedding, sent_lengths = sent_lens, doc_lengths = doc_lens)
                out = self.encoder(inp, sent_lengths = sent_lens, num_sent_per_document = doc_lens, arg=arg)
                out = self.classifier(out)
            else:
                if self.mode == 'multi':
                    inp = inp.reshape((inp.shape[0]*inp.shape[1], inp.shape[2], inp.shape[3]))
                    sent_lens = sent_lens.reshape((sent_lens.shape[0]*sent_lens.shape[1]))
                inp = self.elmo(inp.contiguous())['elmo_representations'][0]
                out = self.encoder(inp, sent_lengths = sent_lens, num_sent_per_document = doc_lens, arg = arg)
                out = self.classifier(out)
        return out



class BiLSTM(nn.Module):
    """
    Baseline BiLSTM to run on the entire document as is:
        1. Option 1: Take the hidden layer output of final cell as the representation of the document
        2. Option 2: Take pool across embedding dimension of all the cells as the representation of the document
    """
    
    def __init__(self, config, max_pool=False):
        super(BiLSTM, self).__init__()
        self.pool = max_pool
        self.embed_dim = config['embed_dim'] if config['embed_dim']==300 else 2*config['embed_dim']
        self.lstm = nn.GRU(self.embed_dim, config["lstm_dim"], bidirectional = True, dropout=config['dropout'], batch_first= True)

    def forward(self, embed, lengths):
        # sorted_len, sorted_idxs = torch.sort(lengths, descending =True)
        # embed = embed[ sorted_idxs ,: , :].to(device)
        sorted_len = lengths
        #print("lengths in each forward = ", sorted_len)
        packed_embed = pack_padded_sequence(embed, sorted_len, batch_first = True, enforce_sorted=False).to(device)
        all_states, hidden_states = self.lstm(packed_embed)
        all_states, _ = pad_packed_sequence(all_states, batch_first = True)
        
        # If not max-pool biLSTM, we extract the h0_l and h0_r from the tuple of tuples 'hn', and concat them to get the final embedding
        if not self.pool:
            out = torch.cat((hidden_states[0][0], hidden_states[0][1]), dim = 1)
            
        # If it is max-pooling biLSTM, set the PADS to very low numbers so that they never get selected in max-pooling
        # Then, max-pool over each dimension(which is now 2D, as 'X' = ALL) to get the final embedding
        elif self.pool:
            # replace PADs with very low numbers so that they never get picked
            out = torch.where(all_states.to('cpu') == 0, torch.tensor(-1e8), all_states.to('cpu'))
            out, _ = torch.max(out, 1)
        # _, unsorted_idxs = torch.sort(sorted_idxs)
        # out = out[unsorted_idxs, :].to(device)
        return out.to(device)



class BiLSTM_reg(nn.Module):
    """
    BiLSTM_regularized : BiLSTM with Temporal Averaging, weight dropout and embedding dropout. Current SOTA on Reuters dataset
    """
    
    def __init__(self, config):
        super(BiLSTM_reg, self).__init__()
        self.tar = 0.0
        self.ar = 0.0
        self.beta_ema = config["beta_ema"]  # Temporal averaging
        self.wdrop = config["wdrop"]  # Weight dropping
        self.embed_droprate = config["embed_drop"]  # Embedding dropouts
        self.dropout = config['dropout']
        self.lstm_dim = config['lstm_dim']
        self.embed_dim = config['embed_dim']
        self.num_classes = config['n_classes']

        self.lstm = nn.LSTM(self.embed_dim, self.lstm_dim, bidirectional = True, dropout=config["dropout"], num_layers=1, batch_first=False).to(device)
        self.classifier = nn.Sequential(nn.Dropout(self.dropout) , nn.Linear(2*self.lstm_dim, 2*self.lstm_dim), nn.ReLU(), nn.Linear(2*self.lstm_dim, self.num_classes-1))

        # Applyying Weight dropout to hh_l0 layer of the LSTM
        weights = ['weight_hh_l0']
        self.lstm = WeightDrop(self.lstm, weights, self.wdrop).to(device)

        if self.beta_ema>0:
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

    def forward(self, inp, embedding, lengths=None):
        sorted_len, sorted_idxs = torch.sort(lengths, descending =True)
        inp = inp.permute(1,0)
        inp = inp[:, sorted_idxs].to(device)

        inp = embedded_dropout(embedding, inp, dropout=self.embed_droprate if self.training else 0) if self.embed_droprate else embedding(inp)
        # print("Input embedding shape = ", inp.shape)

        if lengths is not None:
            inp = torch.nn.utils.rnn.pack_padded_sequence(inp, sorted_len, batch_first=False)
        all_states, _ = self.lstm(inp)

        if lengths is not None:
            all_states,_ = torch.nn.utils.rnn.pad_packed_sequence(all_states, batch_first=False)
            # print("rnn_outs(after lstm) shape = ", rnn_outs.shape)

        out = torch.where(all_states.to('cpu') == 0, torch.tensor(-1e8), all_states.to('cpu'))
        out, _ = torch.max(out, 0)
        _, unsorted_idxs = torch.sort(sorted_idxs)
        out = out[unsorted_idxs, :].to(device)
        out = self.classifier(out)
        return out


    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1-self.beta_ema)*p.data)
    
    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p/(1-self.beta_ema**self.steps_ema))

    def load_params(self, params):
        for p,avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params


    

####################################
## Classes for HAN implementation ##
####################################
"""
The heirarchical attention network working on word and sentence level attention
"""

class HAN(nn.Module):
    def __init__(self, config):
        super(HAN, self).__init__()
        self.word_hidden_dim = config['word_lstm_dim']
        self.embed_dim = config['embed_dim']
        self.num_classes = config['n_classes']
        self.wdrop = config['dropout']

        # Word attention
        self.word_context_weights = nn.Parameter(torch.rand(2*self.word_hidden_dim, 1))
        # self.word_context_weights.data.uniform_(-0.25, 0.25)
        stdv = 1. / math.sqrt(self.word_context_weights.size(0))
        self.word_context_weights.data.normal_(mean=0, std=stdv)
        self.word_attn_gru = nn.GRU(self.embed_dim, self.word_hidden_dim, bidirectional = True, batch_firs=True)
        self.word_lin_projection = nn.Sequential(nn.Linear(2*self.word_hidden_dim, 2*self.word_hidden_dim), nn.Tanh())
        self.word_attn_wts = nn.Softmax()

        # Sentence attention
        self.sent_hidden_dim = config['sent_lstm_dim']
        self.sent_context_weights = nn.Parameter(torch.rand(2*self.sent_hidden_dim, 1))
        self.sent_context_weights.data.uniform_(-0.1, 0.1)
        self.sentence_attn_gru = nn.GRU(2*self.word_hidden_dim, self.sent_hidden_dim, bidirectional = True, batch_first = False)
        self.sent_lin_projection = nn.Sequential(nn.Linear(2*self.sent_hidden_dim, 2*self.sent_hidden_dim), nn.Tanh())
        self.sent_attn_wts = nn.Softmax()

        self.classifier = nn.Linear(2*self.sent_hidden_dim, self.num_classes-1)

    def forward(self, inp, embedding, length):
        inp = inp.permute(1,2,0)
        num_sents = inp.size(0)
        sent_representations = None

        # Word-attention block
        for i in range(num_sents):
            model_inp = inp[i, :]
            model_inp = embedding(model_inp)
            all_states_words, _ = self.word_attn_gru(model_inp)
            out = self.word_lin_projection(all_states_words)
            out = torch.matmul(out, self.word_context_weights)
            out = out.squeeze(dim=2)
            out = self.word_attn_wts(out.transpose(1, 0))
            out = torch.mul(all_states_words.permute(2, 0, 1), out.transpose(1, 0))
            word_attn_outs = torch.sum(out, dim=1).transpose(1, 0).unsqueeze(0)
            if sent_representations is None:
                sent_representations = word_attn_outs
            else:
                sent_representations =  torch.cat((sent_representations, word_attn_outs), dim=0)

        # Sentence-attention Block
        all_states_sents,_ = self.sentence_attn_gru(sent_representations)
        out = self.sent_lin_projection((all_states_sents))
        out = torch.matmul(out, self.sent_context_weights)
        out = out.squeeze(dim=2)
        out = self.sent_attn_wts(out.transpose(1,0))
        out = torch.mul(all_states_sents.permute(2, 0, 1), out.transpose(1, 0))
        out = torch.sum(out, dim=1).transpose(1, 0).unsqueeze(0)
        out = out.squeeze(0)
        sent_attn_out = self.classifier(out)
        return sent_attn_out






class HAN(nn.Module):
    """
    HAN implementation for single as well as multi-GPU training support. 
    - Using pad_and_sort() for the single GPU training
    - Using just_pad() for multi GPU training
    
    NOTE: This implementation of HAN uses dynamic batching to deal with out of memory issues
    """
    
    def __init__(self, config):
        super(HAN, self).__init__()
        self.word_hidden_dim = config['word_lstm_dim']
        self.sent_hidden_dim = config['sent_lstm_dim']
        self.embed_dim = config['embed_dim']
        self.num_classes = config['n_classes']
        self.dropout = config["dropout"]
        self.fc_dim = config['fc_dim']
        self.mode = 'single' if not config['parallel_computing'] else 'multi'

        ### Word attention ###   
        ######################            
        # Initializing the word_context_vector
        self.word_context_vector = nn.Parameter(torch.randn((2 * self.word_hidden_dim, 1)))
        stdv = 1. / math.sqrt(self.word_context_vector.size(0))
        self.word_context_vector.data.normal_(mean=0, std=stdv)
        # Initializing the word LSTM
        self.word_attn_lstm = nn.GRU(2*self.embed_dim, self.word_hidden_dim, bidirectional = True, batch_first= True)       
        # Word level Linear projection layer and attention
        self.word_lin_projection = nn.Sequential(nn.Dropout(config["dropout"]), nn.Linear(2*self.word_hidden_dim, 2*self.word_hidden_dim), nn.Tanh())
        

        ### Sentence attention ###
        ##########################         
        # Initializing the sent_context_vector
        self.sent_context_vector = nn.Parameter(torch.rand(2*self.sent_hidden_dim, 1))
        stdv = 1. / math.sqrt(self.sent_context_vector.size(0))
        self.sent_context_vector.data.normal_(mean=0, std=stdv)      
        # Initializing the sentence LSTM
        self.sentence_attn_lstm = nn.GRU(2*self.word_hidden_dim, self.sent_hidden_dim, bidirectional = True, batch_first= True)       
        # Sentence level Linear projection layer and attention
        self.sent_lin_projection = nn.Sequential(nn.Dropout(config["dropout"]), nn.Linear(2*self.sent_hidden_dim, 2*self.sent_hidden_dim), nn.Tanh())
        # self.classifier = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(2*self.sent_hidden_dim, self.fc_dim), nn.ReLU(), nn.Linear(self.fc_dim, self.num_classes-1))
        
    def forward(self, inp, sent_lengths=None, num_sent_per_document=None, arg = None):
        if self.mode == 'single':
            doc_embedding = self._forward_single(inp, sent_lengths=sent_lengths, num_sent_per_document=num_sent_per_document, recover_idxs = arg)
        else:
            doc_embedding = self._forward_multi(inp, sent_lengths=sent_lengths, num_sent_per_document=num_sent_per_document, max_num_sent = arg)
        
        return doc_embedding
        
        
    
    def _forward_single(self, inp, sent_lengths=None, num_sent_per_document=None, recover_idxs = None):
        batch_size = len(num_sent_per_document)        
        ## Word Level attention block ##
        ################################
        packed_seq = pack_padded_sequence(inp, sent_lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.word_attn_lstm(packed_seq)
        pad_packed_states_word, _ = pad_packed_sequence(out, batch_first=True)
    
        word_pre_attn = self.word_lin_projection(pad_packed_states_word)
        
        # Masked attention
        max_len = pad_packed_states_word.shape[1]
        mask = torch.arange(max_len, device= inp.device)[None, :] < sent_lengths[:, None]
        mask = mask.unsqueeze(2)
        dot_product = word_pre_attn @ self.word_context_vector
        dot_product[~mask] = float('-inf')
        word_attn = torch.softmax(dot_product, dim=1)
        sent_embeddings = torch.sum(word_attn.expand_as(pad_packed_states_word) * pad_packed_states_word, dim=1)
   
        # create new 3d tensor (already padded across dim=1)
        max_num_sent = torch.max(num_sent_per_document)
        sent_embeddings_3d = torch.zeros(batch_size, max_num_sent, sent_embeddings.shape[1]).to(inp.device)

        # fill the 3d tensor
        processed_sent = 0
        for i, num_sent in enumerate(num_sent_per_document):
            sent_embeddings_3d[i, :num_sent.item(), :] = sent_embeddings[processed_sent: processed_sent + num_sent.item(), :]
            processed_sent += num_sent.item()    

        ## Sentence Level attention block ##
        ####################################
        packed_seq = pack_padded_sequence(sent_embeddings_3d, num_sent_per_document, batch_first=True, enforce_sorted = False)
        out, _ = self.sentence_attn_lstm(packed_seq)
        pad_packed_states_sent, _ = pad_packed_sequence(out, batch_first=True)
        sent_pre_attn = self.sent_lin_projection(pad_packed_states_sent)
        
        # Masked attention
        max_len = pad_packed_states_sent.shape[1]
        mask = torch.arange(max_len, device=inp.device)[None, :] < num_sent_per_document[:, None]
        mask = mask.unsqueeze(2)
        dot_product = sent_pre_attn @ self.sent_context_vector
        dot_product[~mask] = float('-inf')
        sent_attn = torch.softmax(dot_product, dim=1)
        doc_embedding = torch.sum(sent_attn.expand_as(pad_packed_states_sent) * pad_packed_states_sent, dim=1)
        # sent_attn = torch.index_select(sorted_sent_attn, dim=0, index=recover_idx_sent).squeeze(2)
        return doc_embedding
    
    
    def _forward_multi(self, inp, sent_lengths=None, num_sent_per_document=None, max_num_sent = None):
        batch_size = len(num_sent_per_document)        
        ## Word Level attention block ##
        ################################
        packed_seq = pack_padded_sequence(inp, sent_lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.word_attn_lstm(packed_seq)
        pad_packed_states_word, _ = pad_packed_sequence(out, batch_first=True)
        
        mask = torch.ones((pad_packed_states_word.shape[0], pad_packed_states_word.shape[1], 2*self.word_hidden_dim)).to(inp.device)
        for i,l in enumerate(sent_lengths):
            if l==1:
                mask[i,:,:] = 0
        pad_packed_states_word = mask*pad_packed_states_word
        
        word_pre_attn = self.word_lin_projection(pad_packed_states_word)
        
        # Masked attention
        max_len = pad_packed_states_word.shape[1]
        mask = torch.arange(max_len, device= inp.device)[None, :] < sent_lengths[:, None]
        mask = mask.unsqueeze(2)
        dot_product = word_pre_attn @ self.word_context_vector
        dot_product[~mask] = float('-inf')
        word_attn = torch.softmax(dot_product, dim=1)
        sent_embeddings = torch.sum(word_attn.expand_as(pad_packed_states_word) * pad_packed_states_word, dim=1)
   
        # create new 3d tensor (already padded across dim=1)
        sent_embeddings_3d = torch.zeros(batch_size, max_num_sent, sent_embeddings.shape[1]).to(inp.device)

        # fill the 3d tensor
        processed_sent = 0
        for i, num_sent in enumerate(num_sent_per_document):
            sent_embeddings_3d[i, :num_sent.item(), :] = sent_embeddings[processed_sent: processed_sent + num_sent.item(), :]
            processed_sent += num_sent.item()    

        ## Sentence Level attention block ##
        ####################################
        packed_seq = pack_padded_sequence(sent_embeddings_3d, num_sent_per_document, batch_first=True, enforce_sorted = False)
        out, _ = self.sentence_attn_lstm(packed_seq)
        pad_packed_states_sent, _ = pad_packed_sequence(out, batch_first=True)
        sent_pre_attn = self.sent_lin_projection(pad_packed_states_sent)
        
        # Masked attention
        max_len = pad_packed_states_sent.shape[1]
        mask = torch.arange(max_len, device=inp.device)[None, :] < num_sent_per_document[:, None]
        mask = mask.unsqueeze(2)
        dot_product = sent_pre_attn @ self.sent_context_vector
        dot_product[~mask] = float('-inf')
        sent_attn = torch.softmax(dot_product, dim=1)
        doc_embedding = torch.sum(sent_attn.expand_as(pad_packed_states_sent) * pad_packed_states_sent, dim=1)
        # sent_attn = torch.index_select(sorted_sent_attn, dim=0, index=recover_idx_sent).squeeze(2)
        return doc_embedding
    
    
    
    
#######################################
##  Class for KimCNN implementation  ##
#######################################        
"""
The (word) CNN based architecture as propsoed by Kim, et.al(2014) 
"""
class Kim_CNN(nn.Module):
    def __init__(self, config):
        super(Kim_CNN, self).__init__()
        self.embed_dim = config['embed_dim'] if config['embed_dim']==300 else 2*config['embed_dim']
        self.num_classes = config["n_classes"]
        self.input_channels = 1
        self.num_kernels = config["kernel_num"]
        self.kernel_sizes = [int(k) for k in config["kernel_sizes"].split(',')]
        self.fc_inp_dim = self.num_kernels * len(self.kernel_sizes)
        self.fc_dim = config['fc_dim']

        self.cnn = nn.ModuleList([nn.Conv2d(self.input_channels, self.num_kernels, (k_size, self.embed_dim)) for k_size in self.kernel_sizes])
        # self.classifier = nn.Sequential(nn.Dropout(config["dropout"]), nn.Linear(self.fc_inp_dim, self.fc_dim), nn.ReLU(), nn.Linear(self.fc_dim, self.num_classes-1))


    def forward(self, inp, embedding = None, lengths=None):
        # x is (B, L, D)
        inp = inp.unsqueeze(1)  # (B, Ci, L, D)
        inp = [F.relu(conv(inp)).squeeze(3) for conv in self.cnn]  # [(B, Co, L), ...]*len(Ks)
        inp = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inp]  # [(B, Co), ...]*len(Ks)
        out = torch.cat(inp, 1) # (B, len(Ks)*Co)
        # out = self.classifier(out)
        return out
    
    

