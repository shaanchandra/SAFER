import os, torch, random, re, torchtext, sys
import numpy as np
import nltk, csv, json
import pandas as pd
from nltk import sent_tokenize, word_tokenize
from tqdm import tqdm
from typing import List, Tuple

from torchtext.data import TabularDataset, Field, NestedField, BucketIterator
import torch.utils.data as data
import warnings
sys.path.append("..")

from allennlp.modules.elmo import batch_to_ids

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


####################################
########  Helper functions  ########
####################################

def clean_string(string):
    # string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    tokens = string.lower().strip().split()
    return tokens

def process_labels(string):
    return [float(x) for x in string]

def process_ids(string):
    # return [str(x) for x in string]
    return str(string)

def split_sents(string):
    string = re.sub(r"[!?]"," ", string)
    return string.strip().split('.')

def clean_string_stop_words_remove(string):
    # string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    nltk_stopwords = nltk.corpus.stopwords.words('english')
    tokens = string.lower().strip().split()
    tokens = [t for t in tokens if not t in nltk_stopwords]
    return tokens
     
        
        
        
#############################################################################

#                        GLove Batching functions                           #

#############################################################################
    


##########################################################
## FakeNewsNet data pre-processing for GLove embeddings ##
##########################################################

class FakeNews(TabularDataset):
    TEXT = Field(sequential = True, batch_first=True, lower=True, use_vocab=True, tokenize=clean_string, include_lengths=True)
    LABEL = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing = process_labels)
    ID = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing = process_ids)
    NUM_CLASSES = 2

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def get_dataset_splits(cls, data_dir, train=os.path.join('train.tsv'),
               validation=os.path.join('val.tsv'),
               test=os.path.join('test.tsv'), **kwargs):
        return super(FakeNews, cls).splits(
            data_dir, train=train, validation=validation, test=test,
            format='tsv', fields=[('text', cls.TEXT), ('label', cls.LABEL), ('id', cls.ID)])

    @classmethod
    def main_handler(cls, config, data_dir, shuffle=True):

        # Getting Data Splits: train, dev, test
        print("\n\n==>> Loading Data splits and tokenizing each document....")
        train, val, test = cls.get_dataset_splits(data_dir)
    
        # Build Vocabulary and obtain embeddings for each word in Vocabulary
        print("\n==>> Building Vocabulary and obtaining embeddings....")
        glove_embeds = torchtext.vocab.Vectors(name= config['glove_path'], max_vectors = int(5e4))
        cls.TEXT.build_vocab(train, val, test, vectors=glove_embeds)

        # Setting 'unk' token as the average of all other embeddings
        if config['model_name'] != 'han':
            cls.TEXT.vocab.vectors[cls.TEXT.vocab.stoi['<unk>']] = torch.mean(cls.TEXT.vocab.vectors, dim=0)

        # Getting iterators for each set
        print("\n==>> Preparing Iterators....")
        train_iter, val_iter, test_iter  = BucketIterator.splits((train, val, test), batch_size=config['batch_size'], repeat=False, shuffle=shuffle,
                                     sort_within_batch=False, device=device)
        return cls.TEXT, cls.LABEL, train_iter, val_iter, test_iter, train, val, test



class FakeNews_HAN(FakeNews):
    NESTING = Field(sequential = True, batch_first=True, lower=True, use_vocab=True, tokenize=clean_string)
    TEXT = NestedField(NESTING, tokenize=sent_tokenize, include_lengths = True)
    LABEL = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing=process_labels)
    ID = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing = process_ids)


class FakeNews_CNN(FakeNews):
    TEXT = Field(sequential = True, batch_first=True, lower=True, use_vocab=True, tokenize=clean_string_stop_words_remove, include_lengths=True)
    LABEL = Field(sequential=False, use_vocab=True, batch_first=True, preprocessing=process_labels)
    ID = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing = process_ids)
    




#########################################################
## FakeHealth data pre-processing for GLove embeddings ##
#########################################################

class FakeHealth(TabularDataset):
    TEXT = Field(sequential = True, batch_first=True, lower=True, use_vocab=True, tokenize=clean_string, include_lengths=True)
    LABEL = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing = process_labels)
    ID = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing = process_ids)
    NUM_CLASSES = 2

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def get_dataset_splits(cls, data_dir, fold, **kwargs):
        # train=os.path.join('train_{}.tsv'.format(fold))
        # validation=os.path.join('val_{}.tsv'.format(fold))
        # test=os.path.join('test_{}.tsv'.format(fold))
        train=os.path.join('train.tsv')
        validation=os.path.join('val.tsv')
        test=os.path.join('test.tsv')
        
        return super(FakeHealth, cls).splits(
            data_dir, train=train, validation=validation, test=test,
            format='tsv', fields=[('text', cls.TEXT), ('label', cls.LABEL), ('id', cls.ID)])

    @classmethod
    def main_handler(cls, config, data_dir, fold, shuffle=True):

        # Getting Data Splits: train, dev, test
        print("\n\n==>> Loading Data splits and tokenizing each document....")
        train, val, test = cls.get_dataset_splits(data_dir, fold)
    
        # Build Vocabulary and obtain embeddings for each word in Vocabulary
        print("\n==>> Building Vocabulary and obtaining embeddings....")
        glove_embeds = torchtext.vocab.Vectors(name= config['glove_path'], max_vectors = int(5e4))
        cls.TEXT.build_vocab(train, val, test, vectors=glove_embeds)

        # Getting iterators for each set
        print("\n==>> Preparing Iterators....")
        train_iter, val_iter, test_iter  = BucketIterator.splits((train, val, test), batch_size=config['batch_size'], repeat=False, shuffle=shuffle,
                                     sort_within_batch=False, device=device)
        return cls.TEXT, cls.LABEL, train_iter, val_iter, test_iter, train, val, test


class FakeHealth_HAN(FakeHealth):
    NESTING = Field(sequential = True, batch_first=True, lower=True, use_vocab=True, tokenize=clean_string)
    TEXT = NestedField(NESTING, tokenize=sent_tokenize, include_lengths = True)
    LABEL = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing=process_labels)
    ID = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing = process_ids)


class FakeHealth_CNN(FakeHealth):
    TEXT = Field(sequential = True, batch_first=True, lower=True, use_vocab=True, tokenize=clean_string_stop_words_remove, include_lengths=True)
    LABEL = Field(sequential=False, use_vocab=True, batch_first=True, preprocessing=process_labels)
    ID = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing = process_ids)
    
    
    
    
    
    
    
#############################################################
## Rumor-veracity data pre-processing for GLove embeddings ##
#############################################################

class Rumor(TabularDataset):
    TEXT = Field(sequential = True, batch_first=True, lower=True, use_vocab=True, tokenize=clean_string, include_lengths=True)
    LABEL = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing = process_labels)
    NUM_CLASSES = 3

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def get_dataset_splits(cls, data_dir, fold, **kwargs):
        if fold==0:
            train=os.path.join('train_1.tsv'.format(fold))
            validation=os.path.join('val.tsv')
            test=os.path.join('val.tsv'.format(fold))
        else:
            train=os.path.join('train_{}.tsv'.format(fold))
            validation=os.path.join('test_{}.tsv'.format(fold))
            test=os.path.join('test_{}.tsv'.format(fold))
        
        return super(FakeNews, cls).splits(
            data_dir, train=train, validation=validation, test=test,
            format='tsv', fields=[('text', cls.TEXT), ('label', cls.LABEL)])

    @classmethod
    def main_handler(cls, config, data_dir, fold, shuffle=True):

        # Getting Data Splits: train, dev, test
        print("\n\n==>> Loading Data splits and tokenizing each document....")
        train, val, test = cls.get_dataset_splits(data_dir, fold)
    
        # Build Vocabulary and obtain embeddings for each word in Vocabulary
        print("\n==>> Building Vocabulary and obtaining embeddings....")
        glove_embeds = torchtext.vocab.Vectors(name= config['glove_path'], max_vectors = int(5e4))
        cls.TEXT.build_vocab(train, val, test, vectors=glove_embeds)

        # Getting iterators for each set
        print("\n==>> Preparing Iterators....")
        train_iter, val_iter, test_iter  = BucketIterator.splits((train, val, test), batch_size=config['batch_size'], repeat=False, shuffle=shuffle,
                                     sort_within_batch=False, device=device)
        return cls.TEXT, cls.LABEL, train_iter, val_iter, test_iter, train, val, test


class Rumor_HAN(Rumor):
    NESTING = Field(sequential = True, batch_first=True, lower=True, use_vocab=True, tokenize=clean_string)
    TEXT = NestedField(NESTING, tokenize=sent_tokenize, include_lengths = True)
    LABEL = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing=process_labels)


class Rumor_CNN(Rumor):
    TEXT = Field(sequential = True, batch_first=True, lower=True, use_vocab=True, tokenize=clean_string_stop_words_remove, include_lengths=True)
    LABEL = Field(sequential=False, use_vocab=True, batch_first=True, preprocessing=process_labels)







#############################################################################

#                        ELMo Batching functions                           #

#############################################################################

#############################
##  ELMo batching for CNN  ##
#############################

def get_elmo_batches(config, n_iter, iters, data, label):
    """
    Elmo requires input as a list of tokens and so can not use the TorchText BucketIterator.
    This function creates batches for Elmo in the required format. 
    """
      
    # To make use of all the training examples while training
    if iters != n_iter:
        start = iters*config['batch_size']
        end = start + config['batch_size']
        batch_text = data[start:end]
        sen_lens = np.array([len(x) for x in batch_text])
        batch_label = label[start:end]
    else:
        # For the final batch, the batch size will be smaller than the batch_size
        start = iters*config['batch_size']
        batch_text = data[start:]
        sen_lens = np.array([len(x) for x in batch_text])
        batch_label = label[start:]    
    
    batch_ids = batch_to_ids(batch_text).to(device)       
    return batch_ids, torch.from_numpy(np.array(batch_label)), torch.from_numpy(sen_lens)




#############################
##  ELMo batching for HAN  ##
#############################
    

class HAN_Dataset(data.Dataset):
    """
    Over-writing the native data.Dataset class to implement Dynamic Batching
    for optimized GPU training.
    """
    def __init__(self, filename, lowercase_sentences=True, max_article_length = 4000):

        self.lowercase_sentences = lowercase_sentences
        self.sent_boundaries = {}
        # Here body_tokens are ALL the docs in a data split which are sentence tokenized
        self._labels, self._all_docs = self._parse_csv_file(filename, max_article_length)
        self._data_size = len(self._labels)
        

    def __getitem__(self, idx):        
        # Taking just 1 document out of all the docs in the split based on the 'idx'
        body_tokens = self._all_docs[idx]
        
        # Get sentence boundaries
        article_indexes = {}
        start_index = 0
        for index, current_sent in enumerate(body_tokens):
            end_index = start_index + len(current_sent)
            article_indexes[index] = (start_index, end_index)
            start_index = end_index

        if self.lowercase_sentences:
            body_tokens = [token.lower() for sentence_tokens in body_tokens for token in sentence_tokens]

        result_embeddings = []
        doc_elmo_ids = batch_to_ids(body_tokens).squeeze(0) # bach_to_ids returns a tensor that is (B x num_words x 50) = (1 x num_words x 50) here
        
        # Get the sentence tokenized structure of documents back using the sentence boundaries
        total_sents = 0
        for i in range(index+1):
            total_sents += 1
            start, end = article_indexes[i]
            len_sent = end-start
            # Limiting the length of a single sentence for memory efficient batching later (splitting longer sentences of a doc into smaller ones)
            if len_sent > 500:
                num_sub_sents = int(np.ceil(len_sent/500))
                for j in range(num_sub_sents-1):
                    total_sents += 1
                    sub_start = start + 500*j
                    sub_end = sub_start + 500
                    result_embeddings.append(doc_elmo_ids[sub_start: sub_end, :])
                result_embeddings.append(doc_elmo_ids[sub_end:end , :])
            else:
                result_embeddings.append(doc_elmo_ids[start:end, :])

        num_tokens_per_sent = [len(sentence_embeddings) for sentence_embeddings in result_embeddings]
        num_of_sents_in_doc = len(result_embeddings)
        
        assert len(num_tokens_per_sent) == num_of_sents_in_doc
        if num_of_sents_in_doc>15:
            num_of_sents_in_doc = 15
            result_embeddings = result_embeddings[:15]
            num_tokens_per_sent = num_tokens_per_sent[:15]

        return result_embeddings, self._labels[idx], num_of_sents_in_doc, num_tokens_per_sent
    
    def __len__(self):
        return self._data_size

    def _parse_csv_file(self, filename, max_article_length):
        '''
        Parses the metaphor CSV file and creates the necessary objects for the dataset
        '''
        df = pd.read_csv(filename, sep='\t', encoding = 'utf8', header=None)
        # Each ele is a document, that we tokenize
        data = [nltk.sent_tokenize(ele[:4000]) for ele in list(df[0])]
        labels = [ele for ele in list(df[1])]
        return labels, data
 



class Dataset_Helper_HAN():
    @staticmethod
    def get_dataset(config, lowercase_sentences = False) -> Tuple[HAN_Dataset, HAN_Dataset]:
        '''
        Parses the data files and creates HAN_Dataset objects
        '''
        train_data_dir = os.path.join(config['data_path'], 'train.tsv')
        val_data_dir = os.path.join(config['data_path'], 'val.tsv')
        test_data_dir = os.path.join(config['data_path'], 'test.tsv')
        # Train
        train_dataset = HAN_Dataset(filename=train_data_dir, lowercase_sentences=lowercase_sentences)        
        # Validation
        val_dataset = HAN_Dataset(filename=val_data_dir, lowercase_sentences=lowercase_sentences)
        # Test    
        test_dataset = HAN_Dataset(filename=test_data_dir, lowercase_sentences=lowercase_sentences)
        return train_dataset, val_dataset, test_dataset
    


class DataLoader_Helper_HAN():
    @classmethod
    def create_dataloaders(cls, train_dataset: data.Dataset = None, validation_dataset: data.Dataset = None, test_dataset: data.Dataset = None, batch_size: int = 32,shuffle: bool = True):
        '''
        Creates DataLoader objects for the given datasets while including padding and sorting
        '''
        # NOTE: We work with batch_szie = 1 since we deploy dynamic batching and thus add docs to batch dynamically based on available memory
        train_loader = None
        if train_dataset:
            train_loader = data.DataLoader(
                dataset=train_dataset,
                batch_size=1,
                num_workers=1,
                shuffle=shuffle)

        validation_loader = None
        if validation_dataset:
            validation_loader = data.DataLoader(
                dataset=validation_dataset,
                batch_size=1,
                num_workers=1,
                shuffle=False)

        test_loader = None
        if test_dataset:
            test_loader = data.DataLoader(
                dataset=test_dataset,
                batch_size=1,
                num_workers=1,
                shuffle=False)
        return train_loader, validation_loader, test_loader
    
    
    
    
class HAN_Batch():
    """
    Creates batches for HAN training with Dynamic batch sizes
    """
    def __init__(self, max_length: int = 15000):
        self._list_of_sequences = []
        self._targets = []
        self._list_of_lengths = []
        self._num_of_sent = []
        self._max_length = max_length

    @property
    def list_of_sequences(self):
        return self._list_of_sequences

    @property
    def targets(self):
        return self._targets

    @property
    def list_of_lengths(self):
        return self._list_of_lengths

    @property
    def num_of_sent(self):
        return self._num_of_sent

    def is_full(self):
        total_sum = sum([sum(lengths) for lengths in self._list_of_lengths])
        return total_sum > self._max_length

    def add_data(self, list_of_sequences, target, num_of_sent, list_of_lengths):
        self._list_of_sequences.append(list_of_sequences)
        self._targets.append(target)
        self._list_of_lengths.append(list_of_lengths)
        self._num_of_sent.append(num_of_sent)
        
        
    def just_pad_batch(self):
        batch_size = len(self._num_of_sent)
        concat_lengths = np.array([length for lengths in self._list_of_lengths for length in lengths])
        concat_sequences = [seq[0] for sequences in self._list_of_sequences for seq in sequences]
        
        max_word_len = np.amax(concat_lengths)
        max_sen_len = max(self._num_of_sent)
        embedding_dimension = concat_sequences[0].shape[2]
        
        padded_sequences = np.zeros((batch_size, max_sen_len, max_word_len, embedding_dimension))
        padded_lengths = np.ones((batch_size,max_sen_len))
        i=0
        for batch in range(batch_size):
            for num_sent in range(self._num_of_sent[batch]):
                padded_sequences[batch, num_sent, :concat_lengths[i], :] = concat_sequences[i].squeeze(1)
                padded_lengths[batch, num_sent] = concat_lengths[i]
                i+=1
        return torch.Tensor(padded_sequences), torch.Tensor(self._targets), torch.LongTensor(self._num_of_sent), torch.LongTensor(padded_lengths)
    

    def pad_batch(self):
        """
        DataLoaderBatch for the HAN should be a list of (sequence, target, length) tuples...
        Returns a padded tensor of sequences sorted from longest to shortest, 
        """
        # concat_lengths - concatted lengths of each sentence - [ batch_size * n_sentences ]
        concat_lengths = np.array([length for lengths in self._list_of_lengths for length in lengths])
        # concat_sequences - the embeddings for each batch for each sentence for each word - [ (batch_size * n_sentences) x n_words x embedding_dim ]
        concat_sequences = [seq[0] for sequences in self._list_of_sequences for seq in sequences]
        max_length = np.amax(concat_lengths)
        
        # # These should be the same list of numbers - checked
        # print("concat lengths =  ", concat_lengths)
        # print([len(seq) for seq in concat_sequences])

        embedding_dimension = concat_sequences[0].shape[2]
        padded_sequences = np.zeros((sum(self._num_of_sent), max_length, embedding_dimension))
        for i, l in enumerate(concat_lengths):
            padded_sequences[i, :l, :] = concat_sequences[i].squeeze(1)           
        return torch.Tensor(padded_sequences), torch.Tensor(self._targets), torch.LongTensor(self._num_of_sent), torch.LongTensor(concat_lengths)
    
    def pad_and_sort_batch(self):
        """
        DataLoaderBatch for the HAN should be a list of (sequence, target, length) tuples...
        Returns a padded tensor of sequences sorted from longest to shortest, 
        """
        # concat_lengths - concatted lengths of each sentence - [ batch_size * n_sentences ]
        concat_lengths = np.array([length for lengths in self._list_of_lengths for length in lengths])
        # concat_sequences - the embeddings for each batch for each sentence for each word - [ (batch_size * n_sentences) x n_words x embedding_dim ]
        concat_sequences = [seq[0] for sequences in self._list_of_sequences for seq in sequences]
        max_length = np.amax(concat_lengths)
        
        # # These should be the same list of numbers - checked
        # print("concat lengths =  ", concat_lengths)
        # print([len(seq) for seq in concat_sequences])

        embedding_dimension = concat_sequences[0].shape[2]
        padded_sequences = np.zeros((sum(self._num_of_sent), max_length, embedding_dimension))
        for i, l in enumerate(concat_lengths):
            padded_sequences[i, :l, :] = concat_sequences[i].squeeze(1)           
        return self._sort_batch(torch.Tensor(padded_sequences), torch.Tensor(self._targets), torch.LongTensor(self._num_of_sent), torch.LongTensor(concat_lengths))


    def _sort_batch(self, batch, targets, num_sentences, lengths):
        """
        Sort a minibatch by the length of the sequences with the longest sequences first
        return the sorted batch targes and sequence lengths.
        This way the output can be used by pack_padded_sequences(...)
        """
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        seq_tensor = batch[perm_idx]

        _, recover_idx = perm_idx.sort(0)
        # print(seq_tensor.shape, targets.shape, recover_idx.shape, num_sentences.shape, seq_lengths.shape)
        return seq_tensor, targets, num_sentences, seq_lengths, recover_idx





#############################################################################

#                        GNN Batching functions                           #

#############################################################################
        
        
def collate_for_lr(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class LR_Dataset(data.Dataset):
    def __init__(self, config, split, seed):

        self.base_dir = os.path.join(os.getcwd(), '..', 'data', 'complete_data', config['data_name'])
        
        if config['data_name'] in ['gossipcop', 'politifact']:
            self.node2id_file = os.path.join(self.base_dir, 'node2id_lr_30_30.json')
            self.doc_embeds_file_gnn = os.path.join(self.base_dir, 'cached_embeds', 'doc_embeds_graph_lr_30_30_{}_{}_{}.pt'.format(split, seed, config['model_name']))
            self.doc_embeds_file_text = os.path.join(self.base_dir, 'cached_embeds', 'doc_embeds_roberta_lr_{}.pt'.format(split))
            # self.doc_embeds_file_text = os.path.join(self.base_dir, 'cached_embeds', 'doc_embeds_gloveavg_lr_{}.pt'.format(split))
            self.split_docs_file = os.path.join(self.base_dir, 'doc_splits_lr.json')
            self._split_docs = json.load(open(self.split_docs_file, 'r'))['{}_docs'.format(split)]
            
        elif config['data_name'] in ['HealthRelease', 'HealthStory']:
            self.node2id_file = os.path.join(self.base_dir, 'node2id_lr_top10.json')
            self.doc_embeds_file_gnn = os.path.join(self.base_dir, 'cached_embeds', 'doc_embeds_graph_top10_lr_test_21_hgt.pt')
            # self.doc_embeds_file_gnn = os.path.join(self.base_dir, 'cached_embeds', 'doc_embeds_graph_top10_lr_{}_{}_{}.pt'.format(split, seed, config['model_name']))
            # self.doc_embeds_file_text = os.path.join(self.base_dir, 'cached_embeds', 'doc_embeds_roberta_21_{}.pt'.format(split))
            self.doc_embeds_file_text = os.path.join(self.base_dir, 'cached_embeds', 'doc_embeds_roberta_{}_{}.pt'.format(seed, split))
            doc2labels_file = os.path.join(os.getcwd(), '..', 'FakeHealth', 'doc2labels_{}.json'.format(config['data_name']))
            doc2labels = json.load(open(doc2labels_file, 'r'))
            self.split_docs_file = os.path.join(os.getcwd(), '..', 'FakeHealth', 'doc_splits_{}.json'.format(config['data_name']))
            self._split_docs = json.load(open(self.split_docs_file, 'r'))['{}_docs'.format(split)]
            
        self.test2id_file = os.path.join(self.base_dir, 'doc2id_encoder.json')
        # self.test2id_file = os.path.join(self.base_dir, 'doc2id_encoder_gloveavg.json')
        self.user_type_file = os.path.join(self.base_dir, 'user_types.json')
        
        if config['data_name'] in ['gossipcop', 'politifact']:
            self.split_docs_labels_file = os.path.join(self.base_dir, 'cached_embeds', 'docs_labels_lr_30_30_{}.json'.format(split))
        else:
            self.split_docs_labels_file = os.path.join(self.base_dir, 'cached_embeds', 'docs_labels_lr_{}.json'.format(split))
            
        self.node2id = json.load(open(self.node2id_file, 'r'))
        self.test2id = json.load(open(self.test2id_file, 'r'))
        self._doc_embeds_gnn = torch.load(self.doc_embeds_file_gnn)
        self.user_types = json.load(open(self.user_type_file, 'r'))
        self._doc_embeds_text = torch.load(self.doc_embeds_file_text, map_location=torch.device('cpu'))
        self.data_size = len(self._split_docs)
        self.mode = config['mode']
        
        if config['data_name'] in ['gossipcop', 'politifact']:
            if not os.path.isfile(self.split_docs_labels_file):
                self.labels = {}
                print("\nCreating labels dict of {} docs...".format(split))
                not_in_either=0
                for split_doc in self._split_docs:
                    real_file = os.path.join(os.getcwd(), '..', 'data', 'base_data', config['data_name'], 'real', str(split_doc)+'.json')
                    fake_file = os.path.join(os.getcwd(), '..', 'data', 'base_data', config['data_name'], 'fake', str(split_doc)+'.json')
                    # print(real_file + "\n" + fake_file)
                    if os.path.isfile(fake_file):
                        label = 1
                    elif os.path.isfile(real_file):
                        label = 0
                    else:
                        # print("[!] WARNING: Did not find {} in either real or fake..".format(split_doc))
                        # sys.exit()
                        not_in_either+=1
                    self.labels[str(split_doc)] = label 
                
                print("not_in_either = ", not_in_either)
                print("Len split_docs_labels = ", len(self.labels))
                print("\nWriting test_doc_labels in: ", self.split_docs_labels_file)
                with open(self.split_docs_labels_file, 'w+') as json_file:
                    json.dump(self.labels, json_file)
            
            else:
                self.labels = json.load(open(self.split_docs_labels_file, 'r'))
                
        else:
            if not os.path.isfile(self.split_docs_labels_file):
                self.labels = {}
                print("\nCreating labels dict of {} docs...".format(split))
                not_in_either=0
                for split_doc in self._split_docs:
                    label = doc2labels[str(split_doc)]
                    self.labels[str(split_doc)] = label 
                
                print("not_in_either = ", not_in_either)
                print("Len split_docs_labels = ", len(self.labels))
                print("\nWriting test_doc_labels in: ", self.split_docs_labels_file)
                with open(self.split_docs_labels_file, 'w+') as json_file:
                    json.dump(self.labels, json_file)
            
            else:
                self.labels = json.load(open(self.split_docs_labels_file, 'r'))
                
      
        

    def __getitem__(self, idx): 
        
        # Taking just 1 document out of all the docs in the split based on the 'idx'
        if self.mode == 'gnn':
            doc = self._split_docs[idx]
            doc_embed = self._doc_embeds_gnn[self.node2id[str(doc)], :]
            label = self.labels[str(doc)]
            return doc_embed, label, doc
        elif self.mode == 'text':
            try:
                doc = self._split_docs[idx]
                # doc_txt = str(doc).split('gossipcop-')[-1]
                # doc_embed = self._doc_embeds_text[self.test2id[str(doc_txt)], :]
                doc_embed = self._doc_embeds_text[self.test2id[str(doc)], :]
                label = self.labels[str(doc)]
            except:
                return None
            return doc_embed, label, doc
        elif self.mode == 'gnn+text':
            try:
                doc = self._split_docs[idx]
                # doc_txt = str(doc).split('gossipcop-')[-1]
                # doc_embed_text = self._doc_embeds_text[self.test2id[str(doc_txt)], :]
                doc_embed_text = self._doc_embeds_text[self.test2id[str(doc)], :]
                doc_embed_gnn = self._doc_embeds_gnn[self.node2id[str(doc)], :]
                doc_embed = torch.cat((doc_embed_text.unsqueeze(1), doc_embed_gnn.unsqueeze(1)))
                label = self.labels[str(doc)]
            except:
                return None
                
            return doc_embed.squeeze(1), label, doc
            
    
    def __len__(self):
        return self.data_size
    
    
    
class LR_Dataset_pheme(data.Dataset):
    def __init__(self, config, split, seed, fold):

        self.base_dir = os.path.join(os.getcwd(), '..', 'data', 'complete_data', 'pheme_cv')
        self.node2id_file = os.path.join(self.base_dir, 'fold_{}'.format(fold), 'node2id_lr.json'.format(config['data_name']))
        self.test2id_file = os.path.join(self.base_dir, 'fold_{}'.format(fold), 'doc2id_encoder.json')
        self.doc_embeds_file_gnn = os.path.join(self.base_dir,'fold_{}'.format(fold), 'doc_embeds_graph_lr_user_{}_{}_{}.pt'.format(split, seed, config['model_name']))
        self.doc_embeds_file_text = os.path.join(self.base_dir, 'fold_{}'.format(fold), 'doc_embeds_roberta_gnn_lr_21_{}.pt'.format(split))
        # self.doc_embeds_file_gnn = os.path.join(self.base_dir, 'doc_embeds_graph_lr_30_noise_{}_{}.pt'.format(split, config['model_name']))
        # self.doc_embeds_file_text = os.path.join(self.base_dir, 'doc_embeds_roberta_lr_{}.pt'.format(split))
        self.split_docs_labels_file = os.path.join(self.base_dir, 'fold_{}'.format(fold), 'doc2labels.json')
        self.split_docs_file = os.path.join(self.base_dir, 'fold_{}'.format(fold), 'doc_splits.json'.format(split))
        self._split_docs = json.load(open(self.split_docs_file, 'r'))['{}_docs'.format(split)]
        self.node2id = json.load(open(self.node2id_file, 'r'))
        self.test2id = json.load(open(self.test2id_file, 'r'))
        self._doc_embeds_gnn = torch.load(self.doc_embeds_file_gnn)
        self._doc_embeds_text = torch.load(self.doc_embeds_file_text, map_location=torch.device('cpu'))
        self.data_size = len(self._split_docs)
        self.mode = config['mode']
        self.labels = json.load(open(self.split_docs_labels_file, 'r'))
      
        

    def __getitem__(self, idx):        
        # Taking just 1 document out of all the docs in the split based on the 'idx'
        if self.mode == 'gnn':
            doc = self._split_docs[idx]
            doc_embed = self._doc_embeds_gnn[self.node2id[str(doc)], :]
            label = self.labels[str(doc)]
            return doc_embed, label
        elif self.mode == 'text':
            try:
                doc = self._split_docs[idx]
                doc_embed = self._doc_embeds_text[self.test2id[str(doc)], :]
                if torch.sum(doc_embed)==0:
                    print("Sum = 0")
                label = self.labels[str(doc)]
            except:
                return None
            return doc_embed, label
        elif self.mode == 'gnn+text':
            try:
                doc = self._split_docs[idx]
                doc_embed_text = self._doc_embeds_text[self.test2id[str(doc)], :]
                doc_embed_gnn = self._doc_embeds_gnn[self.node2id[str(doc)], :]
                doc_embed = torch.cat((doc_embed_text.unsqueeze(1), doc_embed_gnn.unsqueeze(1)))
                label = self.labels[str(doc)]
            except:
                return None
                
            return doc_embed.squeeze(1), label
            
    
    def __len__(self):
        return self.data_size
    



# if __name__ == '__main__':
    # create_data_splits(data_dir= './data/gossipcop.tsv', max_len=4000)
        