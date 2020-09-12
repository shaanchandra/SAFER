import os,re
import sys
import shutil
import json, csv, glob
from tqdm import tqdm
import numpy as np
import random
from sklearn.model_selection import StratifiedShuffleSplit
sys.path.append("..")



def prepare_data_corpus(base_dir = '../FakeHealth'):
    """
    This method reads all the individual JSON files of both the datasets and creates separate .tsv files for each.
    The .tsv file contains the fields: ID, article title, article content and the label
    """
    
    print("\n\n" + "="*50 + "\n\t\tPreparing Data Corpus\n" + "="*50)
    
    datasets = ['HealthRelease', 'HealthStory']
    for dataset in datasets:
        doc2labels = {}
        print("\nCreating doc2labels for:  ", dataset)
        src_dir = os.path.join(base_dir, 'reviews', dataset+'.json')
        doc_labels = json.load(open(src_dir, 'r'))
        for count, doc in enumerate(doc_labels):
            label = 1 if doc['rating'] < 3 else 0  # rating less than 3 is fake
            doc2labels[str(doc['news_id'])] = label
        
        print("Total docs : ", count)
        doc2labels_file = os.path.join(base_dir, 'doc2labels_{}.json'.format(dataset))
        print("\nWriting doc2labels file in :  ", doc2labels_file)
        json.dump(doc2labels, open(doc2labels_file, 'w+'))
    
    for dataset in datasets:
        print("\nCreating the data corpus file for :  ", dataset)
        doc2labels_file = os.path.join(base_dir, 'doc2labels_{}.json'.format(dataset))
        doc2labels = json.load(open(doc2labels_file, 'r'))
        src_dir = os.path.join(base_dir, 'content', dataset+"/*.json")
        final_data_file = os.path.join(os.getcwd(), '..', 'data', dataset+'.tsv')
        all_files = glob.glob(src_dir)
        with open(final_data_file, 'a', encoding = 'utf-8', newline= '') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter='\t')
            csv_writer.writerow(['id', 'title', 'text', 'label'])
            for file in all_files:
                with open(file, 'r') as f:
                    file_content = json.load(f)
                    ID = file.split('\\')[-1]
                    ID = ID.split('.')[0]
                    csv_writer.writerow([ID, file_content['title'], file_content['text'].replace('\n', " "), doc2labels[str(ID)]])
                                    
        print("Final file written in :  ", final_data_file)
        
    return None
    

def create_data_splits_standard(base_dir = os.path.join('../FakeHealth')):
    """
    This method creates train-val-test via random splitting of the dataset in a stratified fashion to ensure similar data distribution
    """
    print("\n\n" + "="*50 + "\n\t\tCreating Data Splits\n" + "="*50)
    
    datasets = ['HealthRelease', 'HealthStory']
    for dataset in datasets:
        print("\nPreparing {} ...".format(dataset))
        src_dir = os.path.join(os.getcwd(), '..', 'data', dataset+'.tsv')
        x_data, y_data, doc_data = [], [], []
        
        # Reading the dataset into workable lists
        removed=0
        lens = []
        with open(src_dir, encoding='utf-8') as data:
            reader = csv.DictReader(data, delimiter='\t')
            for row in reader:
                if isinstance(row['text'], str) and len(row['text']) > 5:
                    text = row['text'].replace('\n', ' ')
                    text = text.replace('\t', ' ')
                    text = re.sub(r'#[\w-]+', 'hashtag', text)
                    text = re.sub(r'https?://\S+', 'url', text)
                    
                    x_data.append(str(text[:5000]))
                    lens.append(len(text[:5000]))
                    y_data.append(int(row['label']))
                    doc_data.append(str(row['id']))
                else:
                    removed+=1
        print("avg lens = ", sum(lens)/len(lens))
        print("max lens = ", max(lens))
        print("minimum lens = ", min(lens))
        print("Total data points removed = ", removed)
        
        # Creating train-val-test split with same/similar label distribution in each split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=21)
        x_rest, x_test, y_rest, y_test = [], [], [], []
        doc_rest, doc_test = [], []
        for train_index, test_index in sss.split(x_data, y_data):
            for idx in train_index:
                x_rest.append(x_data[idx])
                y_rest.append(y_data[idx])
                doc_rest.append(doc_data[idx])
            
            for idx in test_index:
                x_test.append(x_data[idx])
                y_test.append(y_data[idx])  
                doc_test.append(doc_data[idx])
        

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=21)
        for fold, (train_index, val_index) in enumerate(sss.split(x_rest, y_rest)):
            x_train, x_val, y_train, y_val = [], [], [], []
            doc_train, doc_val = [], []
            for idx in train_index:
                x_train.append(x_rest[idx])
                y_train.append(y_rest[idx])
                doc_train.append(doc_rest[idx])
            for idx in val_index:
                x_val.append(x_rest[idx])
                y_val.append(y_rest[idx])
                doc_val.append(doc_rest[idx])                  


        fake, real = get_label_distribution(y_train)
        print("\nFake labels in train split  = {:.2f} %".format(fake*100))
        print("Real labels in train split  = {:.2f} %".format(real*100))
        
        fake, real = get_label_distribution(y_val)
        print("\nFake labels in val split  = {:.2f} %".format(fake*100))
        print("Real labels in val split  = {:.2f} %".format(real*100))
        
        fake, real = get_label_distribution(y_test)
        print("\nFake labels in test split = {:.2f} %".format(fake*100))
        print("Real labels in test split  = {:.2f} %".format(real*100))
        
        print("\nWriting train-val-test files..")
        splits = ['train', 'val', 'test']
        for split in splits:
            if split == 'train':
                x = x_train
                y = y_train
            elif split == 'val':
                x = x_val
                y = y_val
            else:
                x = x_test
                y = y_test
            
            write_dir = os.path.join(os.getcwd(), '..', 'data', dataset)
            if not os.path.exists(write_dir):
                os.makedirs(write_dir)
            write_dir = os.path.join(write_dir, split+'.tsv')
            print("{} file in : {}".format(split, write_dir))
            with open(write_dir, 'a', encoding = 'utf-8', newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter='\t')
                # csv_writer.writerow(['text', 'label'])
                for i in range(len(x)):
                    csv_writer.writerow([x[i], y[i]])

        
        temp_dict = {}
        temp_dict['test_docs'] = doc_test
        temp_dict['train_docs'] = doc_train
        temp_dict['val_docs'] = doc_val
        doc_splits_file = os.path.join(base_dir, 'doc_splits_{}.json'.format(dataset))
        print("Writing doc_splits in : ", doc_splits_file)
        json.dump(temp_dict, open(doc_splits_file, 'w+'))
            
            
            
            
            
            
def create_data_splits_cv(base_dir = './FakeHealth'):
    """
    This method creates cross validaiton splits of the dataset in a stratified fashion to ensure similar data distribution across folds
    """
    
    print("\n\n" + "="*50 + "\n\t\tCreating Data Splits\n" + "="*50)
    
    datasets = ['HealthRelease', 'HealthStory']
    for dataset in datasets:
        print("\nPreparing {} ...".format(dataset))
        doc2labels_file = os.path.join(base_dir, 'doc2labels_{}.json'.format(dataset))
        doc2labels = json.load(open(doc2labels_file, 'r'))
        src_dir = os.path.join(os.getcwd(), '..', 'data', dataset+'.tsv')
        x_data, y_data, doc_data = [], [], []
        
        # Reading the dataset into workable lists
        removed=0
        lens = []
        with open(src_dir, encoding='utf-8') as data:
            reader = csv.DictReader(data, delimiter='\t')
            for row in reader:
                if isinstance(row['text'], str) and len(row['text']) > 5:
                    text = row['text'].replace('\n', ' ')
                    text = text.replace('\t', ' ')
                    text = re.sub(r'#[\w-]+', 'hashtag', text.lower())
                    text = re.sub(r'https?://\S+', 'url', text)
                    
                    x_data.append(str(text[:5000]))
                    lens.append(len(text[:5000]))
                    y_data.append(int(row['label']))
                    doc_data.append(str(row['id']))
                else:
                    removed+=1
        print("avg lens = ", sum(lens)/len(lens))
        print("max lens = ", max(lens))
        print("minimum lens = ", min(lens))
        print("Total data points removed = ", removed)
        
        # Creating train-val-test split with same/similar label distribution in each split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=21)
        x_rest, x_test, y_rest, y_test = [], [], [], []
        doc_rest, doc_test = [], []
        for train_index, test_index in sss.split(x_data, y_data):
            for idx in train_index:
                x_rest.append(x_data[idx])
                y_rest.append(y_data[idx])
                doc_rest.append(doc_data[idx])
            
            for idx in test_index:
                x_test.append(x_data[idx])
                y_test.append(y_data[idx])  
                doc_test.append(doc_data[idx])
        

        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.10, random_state=21)
        for fold, (train_index, val_index) in enumerate(sss.split(x_rest, y_rest)):
            x_train, x_val, y_train, y_val = [], [], [], []
            doc_train, doc_val = [], []
            for idx in train_index:
                x_train.append(x_rest[idx])
                y_train.append(y_rest[idx])
                doc_train.append(doc_rest[idx])
            for idx in val_index:
                x_val.append(x_rest[idx])
                y_val.append(y_rest[idx])
                doc_val.append(doc_rest[idx])                  


            fake, real = get_label_distribution(y_train)
            print("\nFake labels in train split of fold {} = {:.2f} %".format(fold, fake*100))
            print("Real labels in train split of fold {} = {:.2f} %".format(fold, real*100))
            
            fake, real = get_label_distribution(y_val)
            print("\nFake labels in val split of fold {} = {:.2f} %".format(fold, fake*100))
            print("Real labels in val split of fold {} = {:.2f} %".format(fold, real*100))
            
            fake, real = get_label_distribution(y_test)
            print("\nFake labels in test split of fold {} = {:.2f} %".format(fold, fake*100))
            print("Real labels in test split of fold {} = {:.2f} %".format(fold, real*100))
            
            print("\nWriting train-val-test files for fold {}..".format(fold))
            splits = ['train', 'val', 'test']
            for split in splits:
                if split == 'train':
                    x = x_train
                    y = y_train
                elif split == 'val':
                    x = x_val
                    y = y_val
                else:
                    x = x_test
                    y = y_test
                
                write_dir = os.path.join(os.getcwd(), '..', 'data', dataset)
                if not os.path.exists(write_dir):
                    os.makedirs(write_dir)
                write_dir = os.path.join(write_dir, split+'_{}.tsv'.format(fold+1))
                print("{} fold, {} file in : {}".format(fold, split, write_dir))
                with open(write_dir, 'a', encoding = 'utf-8', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file, delimiter='\t')
                    # csv_writer.writerow(['text', 'label'])
                    for i in range(len(x)):
                        csv_writer.writerow([x[i], y[i]])
    
            
            temp_dict = {}
            temp_dict['test_docs'] = doc_test
            temp_dict['train_docs'] = doc_train
            temp_dict['val_docs'] = doc_val
            doc_splits_file = os.path.join(base_dir, 'doc_splits_{}_{}.json'.format(fold+1, dataset))
            json.dump(temp_dict, open(doc_splits_file, 'w+'))

            
   
def get_label_distribution(labels):  
    fake = labels.count(1)
    real = labels.count(0)
    denom = fake+real
    return fake/denom, real/denom
    




if __name__ == '__main__':

    # prepare_data_corpus()
    create_data_splits_standard()
    # get_data_size()