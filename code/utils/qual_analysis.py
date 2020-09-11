import os, torch,json
from collections import defaultdict





def qual_analysis_gossipcop():
    
    doc_embeds_file_text = os.path.join(os.getcwd(), '..', 'data', 'complete_data', 'gossipcop', 'cached_embeds', 'doc_embeds_roberta_lr_test.pt')
    node2id_file = os.path.join(os.getcwd(), '..', 'data', 'complete_data', 'node2id_lr_30_30_gossipcop.json')
    node2id = json.load(open(node2id_file, 'r'))
    test2id_file = os.path.join(os.getcwd(), '..', 'data', 'complete_data', 'gossipcop', 'doc2id_encoder.json')
    test2id = json.load(open(test2id_file, 'r'))
    doc_embeds_text = torch.load(doc_embeds_file_text, map_location=torch.device('cpu'))
    
    
    roberta_docs_file = os.path.join(os.getcwd(), '..', 'data', 'complete_data', 'gossipcop', 'cached_embeds', 'correct_docs_roberta.json')
    gnn_docs_file = os.path.join(os.getcwd(), '..', 'data', 'complete_data', 'gossipcop', 'cached_embeds', 'correct_docs_gcn.json')
    doc2labels_file = os.path.join(os.getcwd(), '..', 'data', 'complete_data', 'gossipcop', 'cached_embeds', 'docs_labels_lr_30_30_test.json')
    
    roberta_docs = json.load(open(roberta_docs_file, 'r'))['correct_preds']
    gnn_docs = json.load(open(gnn_docs_file, 'r'))['correct_preds']
    doc2labels = json.load(open(doc2labels_file, 'r'))
    print(len(doc2labels))
    print(len(roberta_docs), len(gnn_docs))
    gnn_not_roberta = [doc for doc in gnn_docs if doc not in roberta_docs]
    print(len(gnn_not_roberta))
    # print(gnn_not_roberta)
    
    fake_user_counts = json.load(open(os.path.join(os.getcwd(), '..', 'data', 'complete_data', 'gossipcop', 'tsne', 'user_split_counts_30_30.json')))['fake_users']
    real_user_counts = json.load(open(os.path.join(os.getcwd(), '..', 'data', 'complete_data', 'gossipcop', 'tsne', 'user_split_counts_30_30.json')))['real_users']
    doc_user_counts = {}
    for count, doc in enumerate(gnn_not_roberta):
        doc_user_counts[str(doc)] = defaultdict(int)
        file = os.path.join(os.getcwd(), '..', 'data', 'complete_data', 'gossipcop', 'complete', '{}.json'.format(doc))
        users = json.load(open(file, 'r'))['users']
        for user in users:
            if str(user) in fake_user_counts:
                doc_user_counts[str(doc)]['fake_counts']+= fake_user_counts[str(user)]/len(users)
            if str(user) in real_user_counts:
                doc_user_counts[str(doc)]['real_counts']+= real_user_counts[str(user)]/len(users)
        
        print(doc)
        print('Fake' if doc2labels[str(doc)] ==1 else 'Real')
        print('Fake = ', doc_user_counts[str(doc)]['fake_counts'])
        print('Real = ', doc_user_counts[str(doc)]['real_counts'])
        print()
        if count>5:
            break
    
    doc_embed_orig = doc_embeds_text[test2id[str('gossipcop-1883356669')], :]
    sim_scores = {}
    for doc,idx in test2id.items():
        dot = torch.dot(doc_embed_orig, doc_embeds_text[idx, :])
        sim_scores[str(doc)] = dot
    
    c=0
    # sorted_scores = {k: v for k, v in sorted(sim_scores.items(), key=lambda item: item[1])}
    sorted_scores = sorted(sim_scores.items(), key=lambda x: x[1], reverse=True)
    for pair in sorted_scores:
        print(pair[0], pair[1], doc2labels[str(pair[0])])
        c+=1
        if c>10:
            break
        


def qual_analysis_health():
    
    doc_embeds_file_text = os.path.join(os.getcwd(), '..', 'data', 'complete_data', 'HealthStory', 'cached_embeds', 'doc_embeds_roberta_21_test.pt')
    node2id_file = os.path.join(os.getcwd(), '..', 'data', 'complete_data', 'HealthStory', 'node2id_lr_top10.json')
    node2id = json.load(open(node2id_file, 'r'))
    test2id_file = os.path.join(os.getcwd(), '..', 'data', 'complete_data', 'HealthStory', 'doc2id_encoder.json')
    test2id = json.load(open(test2id_file, 'r'))
    doc_embeds_text = torch.load(doc_embeds_file_text, map_location=torch.device('cpu'))
    
    
    roberta_docs_file = os.path.join(os.getcwd(), '..', 'data', 'complete_data', 'HealthStory', 'cached_embeds', 'correct_docs_roberta.json')
    gnn_docs_file = os.path.join(os.getcwd(), '..', 'data', 'complete_data', 'HealthStory', 'cached_embeds', 'correct_docs_gcn.json')
    doc2labels_file = os.path.join(os.getcwd(), '..', 'FakeHealth', 'doc2labels_HealthStory.json')
    
    roberta_docs = json.load(open(roberta_docs_file, 'r'))['correct_preds']
    gnn_docs = json.load(open(gnn_docs_file, 'r'))['correct_preds']
    doc2labels = json.load(open(doc2labels_file, 'r'))
    print(len(doc2labels))
    print(len(roberta_docs), len(gnn_docs))
    gnn_not_roberta = [doc for doc in gnn_docs if doc not in roberta_docs]
    print(len(gnn_not_roberta))
    # print(gnn_not_roberta)
    
    fake_user_counts = json.load(open(os.path.join(os.getcwd(), '..', 'FakeHealth', 'user_split_counts_HealthStory.json')))['fake_users']
    real_user_counts = json.load(open(os.path.join(os.getcwd(), '..', 'FakeHealth', 'user_split_counts_HealthStory.json')))['real_users']
    doc_user_counts = {}
    for count, doc in enumerate(gnn_not_roberta):
        doc_user_counts[str(doc)] = defaultdict(int)
        file = os.path.join(os.getcwd(), '..', 'FakeHealth', 'engagements', 'complete', 'HealthStory', '{}.json'.format(doc))
        users = json.load(open(file, 'r'))['users']
        for user in users:
            if str(user) in fake_user_counts:
                doc_user_counts[str(doc)]['fake_counts']+= fake_user_counts[str(user)]/len(users)
            if str(user) in real_user_counts:
                doc_user_counts[str(doc)]['real_counts']+= real_user_counts[str(user)]/len(users)
                
        
        print(doc)
        print('Fake' if doc2labels[str(doc)] ==1 else 'Real')
        print('Fake = ', doc_user_counts[str(doc)]['fake_counts'])
        print('Real = ', doc_user_counts[str(doc)]['real_counts'])
        print()
        # if count>5:
        #     break
    
    # doc_embed_orig = doc_embeds_text[test2id[str('gossipcop-1883356669')], :]
    # sim_scores = {}
    # for doc,idx in test2id.items():
    #     dot = torch.dot(doc_embed_orig, doc_embeds_text[idx, :])
    #     sim_scores[str(doc)] = dot
    
    # c=0
    # # sorted_scores = {k: v for k, v in sorted(sim_scores.items(), key=lambda item: item[1])}
    # sorted_scores = sorted(sim_scores.items(), key=lambda x: x[1], reverse=True)
    # for pair in sorted_scores:
    #     print(pair[0], pair[1], doc2labels[str(pair[0])])
    #     c+=1
    #     if c>10:
    #         break
        

def qual_analysis_pheme(fold):
    
    doc_embeds_file_text = os.path.join(os.getcwd(), '..', 'data', 'complete_data', 'pheme_cv', 'fold_{}'.format(fold), 'doc_embeds_roberta_lr_filtered_21_test.pt')
    node2id_file = os.path.join(os.getcwd(), '..', 'data', 'complete_data', 'pheme_cv', 'fold_{}'.format(fold), 'node2id_lr_filtered.json')
    node2id = json.load(open(node2id_file, 'r'))
    test2id_file = os.path.join(os.getcwd(), '..', 'data', 'complete_data', 'pheme_cv', 'fold_{}'.format(fold), 'doc2id_encoder_filtered.json')
    user2id_file = os.path.join(os.getcwd(), '..', 'data', 'complete_data', 'pheme_cv', 'fold_{}'.format(fold), 'user2id_train_filtered.json')
    test2id = json.load(open(test2id_file, 'r'))
    user2id = json.load(open(user2id_file, 'r'))
    doc_embeds_text = torch.load(doc_embeds_file_text, map_location=torch.device('cpu'))
    
    
    roberta_docs_file = os.path.join(os.getcwd(), '..', 'data', 'complete_data', 'pheme_cv', 'fold_{}'.format(fold), 'correct_docs_roberta.json')
    gnn_docs_file = os.path.join(os.getcwd(), '..', 'data', 'complete_data', 'pheme_cv', 'fold_{}'.format(fold), 'correct_docs_gat.json')
    doc2labels_file = os.path.join(os.getcwd(), '..', 'data', 'complete_data', 'pheme_cv', 'fold_{}'.format(fold), 'doc2labels.json')
    
    roberta_docs = json.load(open(roberta_docs_file, 'r'))['correct_preds']
    gnn_docs = json.load(open(gnn_docs_file, 'r'))['correct_preds']
    doc2labels = json.load(open(doc2labels_file, 'r'))
    print(len(doc2labels))
    print(len(roberta_docs), len(gnn_docs))
    gnn_not_roberta = [doc for doc in gnn_docs if doc not in roberta_docs]
    print(len(gnn_not_roberta))
    # print(gnn_not_roberta)
    
    # print("\nPreparing user dictionaries...")
    # user_file = os.path.join('data', 'complete_data', 'pheme_cv', 'fold_{}'.format(fold), 'user_fold_counts.json')
    # if not os.path.isfile(user_file):
    #     src_dir = os.path.join('data', 'complete_data', 'pheme_cv', 'complete')
    #     true_users, false_users, unverif_users, all_users = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
    #     for root, dirs, files in os.walk(src_dir):
    #         for count, file in enumerate(files):
    #             doc = file.split('.')[0]
    #             if str(doc) in doc2labels:
    #                 src_file_path = os.path.join(root, file)
    #                 src_file = json.load(open(src_file_path, 'r'))
    #                 # users = src_file['users'][:30]
    #                 user = src_file['source_user']
    #                 if doc2labels[str(doc)] ==1:
    #                     all_users[str(user)] += 1
    #                     true_users[str(user)] += 1
    #                 elif doc2labels[str(doc)] ==0:
    #                     all_users[str(user)] += 1
    #                     false_users[str(user)] += 1
    #                 elif doc2labels[str(doc)] ==2:
    #                     all_users[str(user)] += 1
    #                     unverif_users[str(user)] += 1
    #                 if count % 500 == 0:
    #                         print('{} done..'.format(count))
    #     print("{} True users and {} False users : Total = {}\n".format(len(true_users), len(false_users), len(true_users)+ len(false_users)))
    #     temp_dict = {}
    #     temp_dict['true_users'] = true_users
    #     temp_dict['false_users'] = false_users
    #     temp_dict['unverif_users'] = unverif_users
    #     temp_dict['all_users'] = all_users
    #     print("Writing file in : ", user_file)
    #     with open(user_file, 'w+') as json_file:
    #         json.dump(temp_dict, json_file)
    # else:
    #     print("loading..")
    #     users = json.load(open(user_file, 'r'))
    #     true_users = users['true_users']
    #     false_users = users['false_users']
    #     unverif_users = users['unverif_users']
    #     all_users = users['all_users']
    #     print("{} True users and {} False users : Total = {}\n".format(len(true_users), len(false_users), len(true_users)+ len(false_users)))
    
    
    # only_true, only_false, only_unverif, both = 0,0,0,0
    # for user in all_users:
    #     if str(user) in true_users and str(user) not in false_users and str(user) not in unverif_users:
    #         only_true+=1
    #     if str(user) in false_users and str(user) not in true_users and str(user) not in unverif_users:
    #         only_false+=1
    #     if str(user) in unverif_users and str(user) not in false_users and str(user) not in true_users:
    #         only_true+=1
        
    
    # print("Only_true = ", only_true)
    # print("Only_false = ", only_false)
    # print("Only_unverif = ", only_unverif)
    
    # false_user_counts = json.load(open(os.path.join('data', 'complete_data', 'pheme_cv', 'fold_{}'.format(fold), 'user_fold_counts.json')))['false_users']
    # true_user_counts = json.load(open(os.path.join('data', 'complete_data', 'pheme_cv', 'fold_{}'.format(fold), 'user_fold_counts.json')))['true_users']
    # doc_user_counts = {}
    # for count, doc in enumerate(gnn_not_roberta):
    #     doc_user_counts[str(doc)] = defaultdict(int)
    #     file = os.path.join('data', 'complete_data', 'pheme_cv', 'complete', '{}.json'.format(doc))
    #     user = json.load(open(file, 'r'))['source_user']
    #     if str(user) in false_user_counts:
    #         doc_user_counts[str(doc)]['false_users']+= false_user_counts[str(user)]
    #     if str(user) in true_user_counts:
    #         doc_user_counts[str(doc)]['true_users']+= true_user_counts[str(user)]
                        
    #     print(doc, user)
    #     print('True' if doc2labels[str(doc)] ==1 else 'False')
    #     print('False = ', doc_user_counts[str(doc)]['false_users'])
    #     print('True = ', doc_user_counts[str(doc)]['true_users'])
    #     print()
    #     # if count>5:
    #     #     break
        
    print("\n\nPreparing  INDIVIDUAL user dictionaries...")
    events = ['ch', 'ebola', 'ferg', 'german', 'gurlitt', 'ottawa', 'putin', 'sydney', 'toronto']
    event = events[fold-1]
    print("Event = ", event)
    fold_docs_file = os.path.join(os.getcwd(), '..', 'data', 'complete_data', 'pheme_cv', 'fold_docs_just.json')
    fold_docs = json.load(open(fold_docs_file, 'r'))[str(event)]
    user_file = os.path.join(os.getcwd(), '..', 'data', 'complete_data', 'pheme_cv', 'fold_{}'.format(fold), 'user_fold_counts_indiv.json')
    if not os.path.isfile(user_file):
        src_dir = os.path.join(os.getcwd(), '..', 'data', 'base_data', 'pheme_cv', str(event))
        true_users, false_users, unverif_users, all_users = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
        for root, dirs, files in os.walk(src_dir):
            for count, file in enumerate(files):
                doc = file.split('.')[0]
                if str(doc) in fold_docs:
                    src_file_path = os.path.join(root, file)
                    src_file = json.load(open(src_file_path, 'r'))
                    # users = src_file['users'][:30]
                    user = src_file['user']['id']
                    if doc2labels[str(doc)] ==1:
                        all_users[str(user)] += 1
                        true_users[str(user)] += 1
                    elif doc2labels[str(doc)] ==0:
                        all_users[str(user)] += 1
                        false_users[str(user)] += 1
                    elif doc2labels[str(doc)] ==2:
                        all_users[str(user)] += 1
                        unverif_users[str(user)] += 1
                    if count % 500 == 0:
                            print('{} done..'.format(count))
        print("{} True users, {} False users,  {} Unverif users : Total = {}\n".format(len(true_users), len(false_users), len(true_users)+ len(unverif_users) + len(false_users)))
        temp_dict = {}
        temp_dict['true_users'] = true_users
        temp_dict['false_users'] = false_users
        temp_dict['unverif_users'] = unverif_users
        temp_dict['all_users'] = all_users
        print("Writing file in : ", user_file)
        with open(user_file, 'w+') as json_file:
            json.dump(temp_dict, json_file)
    else:
        print("loading..")
        users = json.load(open(user_file, 'r'))
        true_users = users['true_users']
        false_users = users['false_users']
        unverif_users = users['unverif_users']
        all_users = users['all_users']
        print("{} True users, {} False users,  {} Unverif users : Total = {}\n".format(len(true_users), len(false_users), len(unverif_users), len(true_users)+ len(unverif_users) + len(false_users)))
    
    
    only_true, only_false, only_unverif, mixed, true_false = 0,0,0,0,0
    for user in all_users:
        if str(user) in true_users and str(user) not in false_users and str(user) not in unverif_users:
            only_true+=1
        elif str(user) in false_users and str(user) not in true_users and str(user) not in unverif_users:
            only_false+=1
        if str(user) in unverif_users and str(user) not in false_users and str(user) not in true_users:
            only_unverif+=1
        else:
        # elif str(user) in true_users and str(user) in false_users:
            mixed+=1
   
    print("Only_true = ", only_true)
    print("Only_false = ", only_false)
    print("Only_unverif = ", only_unverif)
    print("true and false = ", true_false)
    print("Mixed = ", mixed)
    
    mixed = 10000000000 if mixed==0 else mixed
    only_true, only_false, only_unverif, true_false = 0,0,0,0
    mixed_avg_true, mixed_avg_false, mixed_avg_unverif = 0,0,0
    for user in all_users:
        if str(user) in true_users and str(user) not in false_users and str(user) not in unverif_users:
            only_true+=1
        if str(user) in false_users and str(user) not in true_users and str(user) not in unverif_users:
            only_false+=1
        if str(user) in unverif_users and str(user) not in false_users and str(user) not in true_users:
            only_unverif+=1
        else:
            if str(user) in true_users:
                mixed_avg_true += true_users[str(user)]/mixed
            if str(user) in false_users:
                mixed_avg_false += false_users[str(user)]/mixed
            if str(user) in unverif_users:
                mixed_avg_unverif += unverif_users[str(user)]/mixed
        
    
    print("\n Mixed_avg_true = ", mixed_avg_true)
    print("Mixed_avg_false = ", mixed_avg_false)
    print("Mixed_avg_unverif = ", mixed_avg_unverif)



if __name__ == '__main__':
    # qual_analysis_gossipcop()
    # qual_analysis_health()
    for fold in [1,3,4,6,8]:
        qual_analysis_pheme(fold = fold)