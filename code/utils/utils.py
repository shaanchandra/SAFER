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
from utils.data_utils_gnn import *
from utils.data_utils_txt import *
from utils.data_utils_hygnn import *

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
        



def calc_elapsed_time(start, end):
    hours, rem = divmod(end-start, 3600)
    time_hours, time_rem = divmod(end, 3600)
    minutes, seconds = divmod(rem, 60)
    time_mins, _ = divmod(time_rem, 60)
    return int(hours), int(minutes), seconds



def evaluation_measures(config, preds, labels):
    f1 = f1_score(labels, preds, average = 'binary', pos_label =1)
    f1_macro = f1_score(labels, preds, average = 'macro')
    recall = recall_score(labels, preds, average = 'binary', pos_label=1)
    precision = precision_score(labels, preds, average = 'binary', pos_label=1)
    accuracy = accuracy_score(labels, preds)
    # print(metrics.classification_report(labels, preds))
    return f1, f1_macro, recall, precision, accuracy



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

        