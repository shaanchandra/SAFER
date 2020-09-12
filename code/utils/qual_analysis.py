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
        





if __name__ == '__main__':
    # qual_analysis_gossipcop()
    # qual_analysis_health()