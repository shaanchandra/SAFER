import os
import sys
import shutil
import json, csv, glob, time, re
from tqdm import tqdm
from pprint import pprint
import numpy as np
import pandas as pd
import random
from scipy.sparse import csr_matrix, lil_matrix, save_npz, load_npz
sys.path.append("..")


        
def convert_annotations(annotation, string = False):
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

        

def creat_base_dir(src_dir):
    labels = []
    for root, dirs, files in os.walk(src_dir):
        for folder in dirs:
            if folder == 'reactions':
                continue
            elif folder == 'source-tweets':
                for count, file in enumerate(files):
                    if file.startswith('.') or file.startswith('structure'):
                        continue
                    src_file_path = os.path.join(root, file)
                    with open(src_file_path, 'r') as j:
                        annotation = json.load(j)
                        labels.append(convert_annotations(annotation, string = False))
    true, false, unverif = get_label_distribution(labels)
    print("\nNo. of labels = ", len(labels))
    print("True labels = {:.2f} %".format(true*100))
    print("False labels = {:.2f} %".format(false*100))
    print("Unverified labels = {:.2f} %".format(unverif*100))
    
    print("\nGetting the source tweets in one file with labels..\n")
    final_data_file = '../data/pheme.tsv'
    c=0
    # getting the source tweets in one file with labels
    with open(final_data_file, 'a+', encoding = 'utf-8') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter='\t')
        csv_writer.writerow(['text', 'label'])
        for root, dirs, files in os.walk(src_dir):
            for folder in dirs:
                if c == len(labels):
                    break
                if folder == 'reactions':
                    continue
                else:
                    src_tweet_file = os.path.join(root, folder, 'source-tweets')
                    src_tweet_file = src_tweet_file + '/{}.json'.format(folder)
                    with open (src_tweet_file, 'r', encoding = 'utf-8') as j:
                        src_tweet= json.load(j)
                    text = src_tweet['text'].replace('\n', ' ')
                    csv_writer.writerow([text, labels[c]])
                    c+=1
                    if c%500 == 0:
                        print("{} done...".format(c))
    return None


  

def create_data_splits(data_dir):
    """
    This method creates train-val-test splits via random splitting
    """
    splits = ['train', 'val', 'test']
    for split in splits:
        print("\n" + "Creating {} split\n".format(split) + "-"*30)
        src_dir = os.path.join(data_dir, split)
        labels = []
        for root, dirs, files in os.walk(src_dir):
            for folder in dirs:
                if folder == 'reactions':
                    continue
                elif folder == 'source-tweets':
                    for count, file in enumerate(files):
                        if file.startswith('.') or file.startswith('structure'):
                            continue
                        src_file_path = os.path.join(root, file)
                        with open(src_file_path, 'r') as j:
                            annotation = json.load(j)
                            labels.append(convert_annotations(annotation, string = False))
        true, false, unverif = get_label_distribution(labels)
        print("\nNo. of labels = ", len(labels))
        print("True labels = {:.2f} %".format(true*100))
        print("False labels = {:.2f} %".format(false*100))
        print("Unverified labels = {:.2f} %".format(unverif*100))
        
        print("\nGetting the source tweets in one file with labels..\n")
        final_data_file = './data/pheme/{}.tsv'.format(split)
        c=0
        # getting the source tweets in one file with labels
        with open(final_data_file, 'a+', encoding = 'utf-8', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter='\t')
            # csv_writer.writerow(['text', 'label'])
            for root, dirs, files in os.walk(src_dir):
                for folder in dirs:
                    if c == len(labels):
                        break
                    if folder == 'reactions':
                        continue
                    else:
                        src_tweet_file = os.path.join(root, folder, 'source-tweets')
                        src_tweet_file = src_tweet_file + '/{}.json'.format(folder)
                        with open (src_tweet_file, 'r', encoding = 'utf-8') as j:
                            src_tweet= json.load(j)
                        text = src_tweet['text'].replace('\n', ' ')
                        text = text.replace('\t', ' ')
                        csv_writer.writerow([text, labels[c]])
                        c+=1
                        if c%500 == 0:
                            print("{} done...".format(c))
    return None




def create_data_folds(src_dir):
    """
    This method creates train-val splits via leave-one-event-out cross validation setup (for all 3 classes)
    """
    
    # Creating the CV folds from the remaining events
    events = ['ch', 'ebola', 'ferg', 'german', 'gurlitt', 'ottawa', 'putin', 'sydney', 'toronto']
    for fold, event in enumerate(events):
        print("\nCreating fold_{}  with  {}  as test set\n".format(fold+1, event) + "-"*50 )
        train_labels, test_labels = [],[]
        test_event = event
        train_events = events.copy()
        train_events.remove(event)
        
        print("Test set: \n" + "-"*20)
        test_data_dir = os.path.join(src_dir, test_event)
        test_data_file = '../data/pheme/test_filtered_{}.tsv'.format(fold+1)
        c=0
        with open(test_data_file, 'a+', encoding = 'utf-8', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter='\t')
            for root, dirs, files in os.walk(test_data_dir):
                for file in files:
                    if file.startswith('.') or file.startswith('structure') or root.endswith('reactions'):
                        continue
                    else:
                        if file.startswith('annotation'):
                            src_file_path = os.path.join(root, file)
                            with open(src_file_path, 'r') as j:
                                annotation = json.load(j)
                                test_labels.append(convert_annotations(annotation, string = False))
                                
                        else:
                            src_tweet_file = os.path.join(root, file)
                            with open (src_tweet_file, 'r', encoding = 'utf-8') as j:
                                src_tweet= json.load(j)
                            text = src_tweet['text'].replace('\n', ' ')
                            text = text.replace('\t', ' ')
                            csv_writer.writerow([text, test_labels[c]])
                            c+=1
                            if c%100 == 0:
                                print("{} done...".format(c))
        true, false, unverif = get_label_distribution(test_labels)
        print("\nTotal test instances = ", len(test_labels))
        print("True test labels = {:.2f} %".format(true*100))
        print("False test labels = {:.2f} %".format(false*100))
        print("Unverified test labels = {:.2f} %".format(unverif*100))
        
        print("\nTrain set: \n" + "-"*20)
        train_data_file = '../data/pheme/train_{}.tsv'.format(fold+1)
        c=0
        for train_event in train_events:
            train_data_dir = os.path.join(src_dir, train_event)
            with open(train_data_file, 'a+', encoding = 'utf-8', newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter='\t')
                for root, dirs, files in os.walk(train_data_dir):
                    for file in files:
                        if file.startswith('.') or file.startswith('structure') or root.endswith('reactions'):
                            continue
                        else:
                            if file.startswith('annotation'):
                                src_file_path = os.path.join(root, file)
                                with open(src_file_path, 'r') as j:
                                    annotation = json.load(j)
                                    train_labels.append(convert_annotations(annotation, string = False))
                                    
                            else:
                                src_tweet_file = os.path.join(root, file)
                                with open (src_tweet_file, 'r', encoding = 'utf-8') as j:
                                    src_tweet= json.load(j)
                                text = src_tweet['text'].replace('\n', ' ')
                                text = text.replace('\t', ' ')
                                csv_writer.writerow([text, train_labels[c]])
                                c+=1
                                if c%1000 == 0:
                                    print("{} done...".format(c))
        true, false, unverif = get_label_distribution(train_labels)
        print("\nTotal train instances = ", len(train_labels))
        print("True train labels = {:.2f} %".format(true*100))
        print("False train labels = {:.2f} %".format(false*100))
        print("Unverified train labels = {:.2f} %".format(unverif*100))
    return None




def create_data_folds_filtered(src_dir):
    """
    This method creates train-val splits via leave-one-event-out cross validation setup (for 2 classes -- unverified label dropped)
    """
    
    # Creating the CV folds from the remaining events
    events = ['ch', 'ebola', 'ferg', 'german', 'gurlitt', 'ottawa', 'putin', 'sydney', 'toronto']
    
    for fold, event in enumerate(events):
        print("\nCreating fold_{}  with  {}  as test set\n".format(fold+1, event) + "-"*50 )
        train_labels, test_labels = [],[]
        test_event = event
        train_events = events.copy()
        train_events.remove(event)
        labels_dict_file = os.path.join('data', 'complete_data', 'pheme_cv', 'fold_{}'.format(fold+1), 'doc2labels_filtered.json')
        doc2labels = json.load(open(labels_dict_file, 'r'))
        
        print("Test set: \n" + "-"*20)
        test_data_dir = os.path.join(src_dir, test_event)
        test_data_file = './data/pheme/test_filtered_{}.tsv'.format(fold+1)
        c=0
        with open(test_data_file, 'a+', encoding = 'utf-8', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter='\t')
            for root, dirs, files in os.walk(test_data_dir):
                for file in files:
                    if file.startswith('.') or file.startswith('structure') or root.endswith('reactions'):
                        continue
                    else:
                        doc = file.split('.')[0]
                        if str(doc) in doc2labels:
                            test_labels.append(doc2labels[str(doc)])
                            src_tweet_file = os.path.join(root, file)
                            with open (src_tweet_file, 'r', encoding = 'utf-8') as j:
                                src_tweet= json.load(j)
                            text = src_tweet['text'].replace('\n', ' ')
                            text = text.replace('\t', ' ')
                            csv_writer.writerow([text, test_labels[c]])
                            c+=1
                            if c%100 == 0:
                                print("{} done...".format(c))
        true, false, unverif = get_label_distribution(test_labels)
        assert len(test_labels) == c
        print("\nTotal test instances = ", len(test_labels))
        print("True test labels = {:.2f} %".format(true*100))
        print("False test labels = {:.2f} %".format(false*100))
        print("Unverified test labels = {:.2f} %".format(unverif*100))
        
        print("\nTrain set: \n" + "-"*20)
        train_data_file = './data/pheme/train_filtered_{}.tsv'.format(fold+1)
        c=0
        for train_event in train_events:
            train_data_dir = os.path.join(src_dir, train_event)
            with open(train_data_file, 'a+', encoding = 'utf-8', newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter='\t')
                for root, dirs, files in os.walk(train_data_dir):
                    for file in files:
                        if file.startswith('.') or file.startswith('structure') or root.endswith('reactions'):
                            continue
                        else:
                            doc = file.split('.')[0]
                            if str(doc) in doc2labels:
                                train_labels.append(doc2labels[str(doc)])
                                src_tweet_file = os.path.join(root, file)
                                with open (src_tweet_file, 'r', encoding = 'utf-8') as j:
                                    src_tweet= json.load(j)
                                text = src_tweet['text'].replace('\n', ' ')
                                text = text.replace('\t', ' ')
                                csv_writer.writerow([text, train_labels[c]])
                                c+=1
                                if c%100 == 0:
                                    print("{} done...".format(c))
        true, false, unverif = get_label_distribution(train_labels)
        print("\nTotal train instances = ", len(train_labels))
        print("True train labels = {:.2f} %".format(true*100))
        print("False train labels = {:.2f} %".format(false*100))
        print("Unverified train labels = {:.2f} %".format(unverif*100))
    return None




#########################
##  For Crawling jobs  ##
#########################

def create_tweet_retweet_folder(src_dir):
    print("\nCreating Tweets folder...")
    c=0
    dest_path = './pheme_final/pheme_crawl/tweets'
    for root, dirs, files in os.walk(src_dir):
        for folder in dirs:
            if c == 2402:
                break
            if folder == 'reactions':
                continue
            else:
                src_tweet_file = os.path.join(root, folder, 'source-tweets')
                src_tweet_file = src_tweet_file + '/{}.json'.format(folder)
                shutil.copy(src_tweet_file, dest_path)
                c+=1
                if c%500 == 0:
                    print("{} done...".format(c))
                    
    print("\nCreating Re-tweets folder...")
    c=0
    retweets=0
    dest_path = './pheme_final/pheme_crawl/retweets'
    for root, dirs, files in os.walk(src_dir):
        for folder in dirs:
            if c == 2402:
                break
            if folder != 'reactions':
                continue
            else:
                src_dir2 = os.path.join(root, folder)
                path = dest_path+'\\'+src_dir2.split('\\')[-2]+'.json'
                for root2, dirs2, files2 in os.walk(src_dir2):
                    for file in files2:
                        if file.startswith('.'):
                            continue
                        else:
                            src_file = os.path.join(root2, file)
                            with open (src_file, 'r', encoding = 'utf-8') as j:
                                src_file = json.load(j)
                            with open(path, 'a+') as outfile:
                                outfile.write(json.dumps(src_file))
                                outfile.write("\n")
                            retweets+=1
            c+=1
            if c%500==0:
                print("{} done..".format(c))
    print("\nC = ", c)
    print("Retweets = ", retweets)

    return None



def create_aggregate_folder(data_dir):
    print("\nCreating aggregate files for all docs and their users......")
    parts = ['tweets', 'retweets']
    print("\n" + "-"*60 + "\n \t\t Analyzing PHEME dataset\n" + '-'*60)
    for part in parts:
        print("\nIterating over : ", part)
        start = time.time()
        src_dir = os.path.join(data_dir, part)
        dest_dir = os.path.join(data_dir, 'complete')
        if not os.path.exists(dest_dir):
            print("Creating dir:  {}\n".format(dest_dir))
            os.makedirs(dest_dir)
        if part == 'tweets':
            for root, dirs, files in os.walk(src_dir):
                for count, file in enumerate(files):
                    src_file_path = os.path.join(root, file)
                    dest_file_path = os.path.join(dest_dir, file)
                    
                    with open(src_file_path, 'r') as j:
                        src_file = json.load(j)            
                    user_id = src_file['user']['id']
                    with open(dest_file_path, 'w+', encoding = 'utf-8', newline='') as csv_file:
                        csv_writer = csv.writer(csv_file)
                        csv_writer.writerow(['user_id'])
                        csv_writer.writerow([user_id])
                    if count%500 == 0:
                        # print("{} done".format(count), end='\r')
                        print("{} done".format(count))
        elif part == 'retweets':
            if not os.path.exists(dest_dir):
                print("Creating dir   {}", dest_dir)
                os.makedirs(dest_dir)
            
            for root, dirs, files in os.walk(src_dir):
                for count,file in enumerate(files):
                    src_file_path = os.path.join(root, file)
                    dest_file_path = os.path.join(dest_dir, file)
                    with open(src_file_path, encoding= 'utf-8', newline = '') as csv_file:
                        lines = csv_file.readlines()
                        user_ids = []
                        for line in lines:
                            file = json.loads(line)
                            user_ids.append(file['user']["id"])
                        user_ids = list(set(user_ids))
                        with open(dest_file_path, 'w+', encoding = 'utf-8', newline='') as dest_file:
                            dest_csv_writer = csv.writer(dest_file)
                            for u in user_ids:
                                dest_csv_writer.writerow([u])
                    if count%500 == 0:
                        # print("{}  done".format(count), end='\r')
                        print("{} done".format(count))


     
def get_label_distribution(labels):  
    true = labels.count(1)
    false = labels.count(0)
    unverified = labels.count(2)
    denom = true + false + unverified  
    if denom==0:
        denom=1
    return true/denom, false/denom, unverified/denom



if __name__ == '__main__':
    # creat_base_dir(src_dir = '../pheme_final')
    # create_data_splits('../data/base_data/pheme')
    # create_tweet_retweet_folder(src_dir = '../data/base_data/pheme/all')
    # create_aggregate_folder(data_dir= '../pheme_final/pheme_crawl')
    # create_data_folds(src_dir = '../data/base_data/pheme_cv')
    create_data_folds_filtered(src_dir = '../data/base_data/pheme_cv')