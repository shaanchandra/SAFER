import os
import sys
import shutil
import json, csv, glob
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import random, re
sys.path.append("..")


def create_base_directory():
    """
    The crawled folder has files divided by labels and dataset and all individual content in their respective folders.
    This script takes them from their respective folders and moves them to a common folder for easier processing later on.
    """
    labels = ['real', 'fake']
    dataset = ['gossipcop', 'politifact']
    for data in dataset:
        for label in labels:
            print("\n --> Processing :  {} dataset ({} labels)".format(data, label))
            
            src_path = os.path.join('../new/FakeNewsNet/code/fakenewsnet_dataset', data)
            src_path = os.path.join(src_path, label)
            
            dest_path = os.path.join('base_data', data)
            dest_path = os.path.join(dest_path, label)
        
            
            # Renaming the JSON files as unique IDs
            for root, dirs, files in os.walk(src_path):
                for file in files:
                    orig_file_path = os.path.join(root, file)   
                    folder_name = orig_file_path.split("\\")[-2]
                    if folder_name == 'tweets'or folder_name == 'retweets':
                        continue
                    else:
                        # print("Original file path  =    ", orig_file_path)
                        # print("Folder name  =    ", folder_name)
                        new_file_name = "/"+folder_name+".json"
                        os.rename(orig_file_path, root+new_file_name)
                        
            # Moving files from individual folders to one central folder
            file_list = []
            for root, dirs, files in os.walk(src_path):
                for file in files:
                    orig_file_path = os.path.join(root, file)
                    folder_name = orig_file_path.split("\\")[-2]
                    if folder_name == 'tweets' or folder_name == 'retweets':
                        continue
                    else:
                        file_list.append(orig_file_path)
            for f in file_list:
                shutil.move(f, dest_path)   
            print("DONE")
    return None




def prepare_data_corpus(new_dir= os.path.join(os.getcwd(), '..', 'data', 'base_data')):
    """
    This method reads all the individual JSON files of both the datasets and creates separate .tsv files for each.
    The .tsv file contains the fields: ID, article title, article content and the label
    """
    labels = ['real', 'fake']
    dataset = ['gossipcop', 'politifact']
    for data in dataset:
        c=0
        doc2id = {}
        final_data_file = os.path.join(os.getcwd(), '..', 'data', data+'.tsv')
 
        for label in labels:
            print("\n --> Processing :  {} dataset ({} labels)".format(data, label))
            data_label = 1 if label=='fake' else 0
            print(data_label)
            DATA_DIR = os.path.join(new_dir, data, label)
            DATA_DIR = DATA_DIR+"/*.json"

            all_files = glob.glob(DATA_DIR)
            with open(final_data_file, 'a', encoding = 'utf-8') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter='\t')
                csv_writer.writerow(['id', 'title', 'text', 'label'])
                for file in all_files:
                    with open(file, 'r') as f:
                        file_content = json.load(f)
                        if data == 'politifact':                        
                            ID = file.split('politifact')[2].replace('.json','')
                        else:
                            # ID = file.split('\\')[-1]
                            # ID = ID.split('.')[0]
                            ID = file.split('gossipcop-')[1].replace('.json', '')
                            doc2id[str(ID)] = c
                            c+=1
                        csv_writer.writerow([ID, file_content['title'],file_content['text'], data_label])
            print("Done: ", label)
        # doc2id_file = os.path.join('complete_data', data, 'doc2id_encoder.json')
        # print("Saving doc2id_glove in :", doc2id_file)
        # with open(doc2id_file, 'w+') as v:
        #     json.dump(doc2id, v)
    return None




def create_data_splits(max_len=5000):
    """
    This method creates train-val-test via random splitting of the dataset in a stratified fashion to ensure similar data distribution
    """
    print("\n\n" + "="*50 + "\n\t\tCreating Data Splits\n" + "="*50)
    
    datasets = ['gossipcop', 'politifact']
    for dataset in datasets:
        print("\nPreparing {} ...".format(dataset))
        src_dir = os.path.join(os.getcwd(), '..', 'data', dataset+'.tsv')
        x_data, y_data, doc_id = [], [], []
        
        # Reading the dataset into workable lists
        removed=0
        lens = []
        with open(src_dir, encoding='utf-8') as data:
            reader = csv.DictReader(data, delimiter='\t')
            for row in reader:
                if isinstance(row['text'], str) and len(row['text']) > 25:
                    text = row['text'].replace('\n', ' ')
                    text = text.replace('\t', ' ')
                    text = re.sub(r'#[\w-]+', 'hashtag', text)
                    text = re.sub(r'https?://\S+', 'url', text)
                    
                    x_data.append(str(text[:max_len]))
                    lens.append(len(text[:max_len]))
                    y_data.append(int(row['label']))
                    doc_id.append(str(row['id']))
                else:
                    removed+=1
        print("avg lens = ", sum(lens)/len(lens))
        print("max lens = ", max(lens))
        print("minimum lens = ", min(lens))
        print("Total data points removed = ", removed)
        
        # Creating train-val-test split with same/similar label distribution in each split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=21)
        x_rest, x_test, y_rest, y_test = [], [], [], []
        doc_id_rest, doc_id_test = [], []
        for train_index, test_index in sss.split(x_data, y_data):
            for idx in train_index:
                x_rest.append(x_data[idx])
                y_rest.append(y_data[idx])
                doc_id_rest.append(doc_id[idx])
            
            for idx in test_index:
                x_test.append(x_data[idx])
                y_test.append(y_data[idx])  
                doc_id_test.append(doc_id[idx])
        

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=21)
        for fold, (train_index, val_index) in enumerate(sss.split(x_rest, y_rest)):
            x_train, x_val, y_train, y_val = [], [], [], []
            doc_id_train, doc_id_val = [], []
            for idx in train_index:
                x_train.append(x_rest[idx])
                y_train.append(y_rest[idx])
                doc_id_train.append(doc_id_rest[idx])
            for idx in val_index:
                x_val.append(x_rest[idx])
                y_val.append(y_rest[idx])
                doc_id_val.append(doc_id_rest[idx])                  


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
                id_list = doc_id_train
            elif split == 'val':
                x = x_val
                y = y_val
                id_list = doc_id_val
            else:
                x = x_test
                y = y_test
                id_list = doc_id_test
            
            write_dir = os.path.join(os.getcwd(), '..', 'data', dataset)
            if not os.path.exists(write_dir):
                os.makedirs(write_dir)
            write_dir = os.path.join(write_dir, split+'.tsv')
            print("{} file in : {}".format(split, write_dir))
            with open(write_dir, 'a', encoding = 'utf-8', newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter='\t')
                # csv_writer.writerow(['text', 'label'])
                for i in range(len(x)):
                    csv_writer.writerow([x[i], y[i], id_list[i]])

        
        temp_dict = {}
        temp_dict['test_docs'] = doc_id_test
        temp_dict['train_docs'] = doc_id_train
        temp_dict['val_docs'] = doc_id_val
        doc_splits_file = os.path.join(os.getcwd(), '..', 'data', dataset, 'doc_splits.json')
        print("Writing doc_splits in : ", doc_splits_file)
        json.dump(temp_dict, open(doc_splits_file, 'w+'))




def get_label_distribution(labels):  
    fake = labels.count(1)
    real = labels.count(0)
    denom = fake+real
    return fake/denom, real/denom



def get_data_size():
    
    src_doc_dir = os.path.join(os.getcwd(), '..', 'data', 'base_data', 'gossipcop')
    count, total, fake, real = 0,0,0,0
    small = []
    lengths = []
    for root, dirs, files in os.walk(src_doc_dir):
        for file in files:
            src_file_path = os.path.join(root, file)
            with open(src_file_path, 'r') as f:
                total+=1
                file_content = json.load(f)
                text = file_content['text'].lower()
                lengths.append(len(text))
                if len(text) > 25:
                    split = root.split('\\')[-1]
                    if split == 'fake':
                        fake+=1
                    else:
                        real+=1
                    doc = file.split('.')[0]
                    small.append(doc)
                    count+=1
    
    print(count, total)
    print(fake, real)
    print(max(lengths))
    print(sum(lengths) / len(lengths))
    print(small[:10])
    




if __name__ == '__main__':
    
    # create_base_directory()
    # prepare_data_corpus()
    create_data_splits()
    # get_data_size()