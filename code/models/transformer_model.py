from __future__ import absolute_import, division, print_function

import random

import torch, math, os, sys
import numpy as np
from torch.nn import Parameter
from torchnlp.nn import WeightDrop
from functools import wraps
from copy import deepcopy
from torch import nn
import torch.utils.data as data
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from simpletransformers.experimental.classification import classification_model
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

import warnings
warnings.filterwarnings("ignore")
sys.path.append("..")

# from utils.utils import *
from multiprocessing import cpu_count

import torch

from scipy.stats import pearsonr
from sklearn.metrics import (
    mean_squared_error,
    matthews_corrcoef,
    confusion_matrix,
    label_ranking_average_precision_score,
)
from tensorboardX import SummaryWriter
from tqdm.auto import trange, tqdm

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from torch.nn.utils.rnn import pad_sequence

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import (
    WEIGHTS_NAME,
    BertConfig,
    BertTokenizer,
    XLNetConfig,
    XLNetTokenizer,
    XLMConfig,
    XLMTokenizer,
    RobertaConfig,
    RobertaTokenizer,
    DistilBertConfig,
    DistilBertTokenizer,
    AlbertConfig,
    AlbertTokenizer,
    CamembertConfig,
    CamembertTokenizer,
)

from simpletransformers.experimental.classification.classification_utils import (
    InputExample,
    convert_examples_to_features,
)


from transformers.modeling_bert import BertPreTrainedModel, BertModel
from transformers.modeling_roberta import (
    RobertaConfig,
    RobertaModel,
    ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
    BertPreTrainedModel,
)
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    


class BertForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """

    def __init__(self, config, weight=None, sliding_window=False, extract_embeddings = False):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.weight = weight
        self.sliding_window = sliding_window
        self.extract_embeddings = extract_embeddings
        
        if self.extract_embeddings:
            # self.attn = nn.Parameter(torch.zeros(768,1), requires_grad = True)
            self.attn = nn.Linear(768,1)
            # stdv = 1. / math.sqrt(self.attn.size(0))
            # self.attn.data.normal_(mean=0, std=stdv)
            nn.init.xavier_normal(self.attn.weight)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None,):

        all_outputs = []
        if self.sliding_window:
            # input_ids is really the list of inputs for each "sequence window"
            labels = input_ids[0]["labels"]
            for inputs in input_ids:
                ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                token_type_ids = inputs["token_type_ids"]
                outputs = self.bert(
                    ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                )
                if self.extract_embeddings:
                    # all_outputs.append(torch.mean(outputs[0], axis=1))  # (B,L,D) --> (B,D)
                    all_outputs.append(outputs[0][:, 0, :])   # (B,L,D) --> (B,D)
                else:
                    all_outputs.append(outputs[1]) # [B,D]
            
        else:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )
            
            pooled_output = outputs[1]  # [B,D]
        
        if (not self.extract_embeddings) and (self.sliding_window):
            # if normal sliding window aggregation on the whole doc
            pooled_output = torch.mean(torch.stack(all_outputs), axis=0)
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
        elif (not self.extract_embeddings) and (not self.sliding_window):
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
        
        else:
            # If using attention on the doc features produced by model
            pooled_output = torch.stack(all_outputs, dim=1)  # Stack all the individual chunks  [B, L, D]
            attention_score = self.attn(pooled_output)  # [B, L, 1]
            attention_score = F.softmax(attention_score).view(pooled_output.size(0), pooled_output.size(1), 1)
            weighted_out = pooled_output * attention_score # [B, L, D]
            weighted_out = torch.sum(weighted_out, axis=1) # [B,D]
            weighted_out = self.dropout(weighted_out)
            logits = self.classifier(weighted_out)
        
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss(weight=self.weight)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, extract_embeddings):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.extract_embeddings = extract_embeddings

    def forward(self, features, **kwargs):
        # features is [B,L,D]
        if not self.extract_embeddings:
            x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        else:
            x = torch.sum(features, axis=1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    


class CNN_model(nn.Module):
    def __init__(self, config):
        super(CNN_model, self).__init__()
        self.embed_dim = 768 
        self.input_channels = 1
        self.num_kernels = 128  # config["kernel_num"]
        self.kernel_sizes = [3,4,5]  # [int(k) for k in config["kernel_sizes"].split(',')]
        self.fc_inp_dim = self.num_kernels * len(self.kernel_sizes)
        # self.fc_dim = config['fc_dim']

        self.cnn = nn.ModuleList([nn.Conv2d(self.input_channels, self.num_kernels, (k_size, self.embed_dim)) for k_size in self.kernel_sizes])
        self.classifier = nn.Sequential(nn.Dropout(config.hidden_dropout_prob), 
                                            nn.Linear(self.fc_inp_dim, 64), 
                                            nn.ReLU(), 
                                            nn.Linear(64, config.num_labels))

    def forward(self, inp, embedding = None, lengths=None):
        # x is (B, L, D)
        inp = inp.unsqueeze(1)  # (B, Ci, L, D)
        inp = [F.relu(conv(inp)).squeeze(3) for conv in self.cnn]  # [(B, Co, L), ...]*len(Ks)
        inp = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inp]  # [(B, Co), ...]*len(Ks)
        out = torch.cat(inp, 1) # (B, len(Ks)*Co)
        out = self.classifier(out)
        return out    



class RobertaForSequenceClassification(BertPreTrainedModel):
   
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "roberta"

    def __init__(self, config, weight=None, sliding_window=False, extract_embeddings = False, classif_type='mlp'):
        super(RobertaForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.classif_type = classif_type
        print("Classifier type = ", self.classif_type)

        self.roberta = RobertaModel(config)
        self.weight = weight
        self.sliding_window = sliding_window
        self.extract_embeddings = extract_embeddings
        
        if self.extract_embeddings:
            self.attn = nn.Linear(1024,1)
            # stdv = 1. / math.sqrt(self.attn.size(0))
            # self.attn.data.normal_(mean=0, std=stdv)
            nn.init.xavier_normal(self.attn.weight)
        if self.classif_type != 'cnn':
            self.classifier = RobertaClassificationHead(config, self.extract_embeddings)
        else:
            self.classifier = CNN_model(config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,labels=None,):
        all_outputs = []
        if self.sliding_window:
            # input_ids is really the list of inputs for each "sequence window"
            labels = input_ids[0]["labels"]
            for inputs in input_ids:
                ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                token_type_ids = inputs["token_type_ids"]
                outputs = self.roberta(
                    ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                )
                if self.extract_embeddings:
                    # all_outputs.append(torch.mean(outputs[0], axis=1))  # (B,L,D) --> (B,D)
                    all_outputs.append(outputs[0][:, 0, :])
                else:
                    all_outputs.append(outputs[0])
            
        else:
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )
            sequence_output = outputs[0]
        if self.classif_type == 'cnn':
            if self.sliding_window:
                sequence_output = torch.mean(torch.stack(all_outputs), axis=0)
            logits = self.classifier(sequence_output)
         
        else:
            if (not self.extract_embeddings) and (self.sliding_window):
                # if normal sliding window aggregation on the whole doc
                sequence_output = torch.mean(torch.stack(all_outputs), axis=0)
                logits = self.classifier(sequence_output)
            elif (not self.extract_embeddings) and (not self.sliding_window):
                logits = self.classifier(sequence_output)
            
            else:
                # If using attention on the doc features produced by model
                sequence_output = torch.stack(all_outputs, dim=1)  # Stack all the individual chunks
                attention_score = self.attn(sequence_output)
                attention_score = F.softmax(attention_score).view(sequence_output.size(0), sequence_output.size(1), 1)
                weighted_out = sequence_output * attention_score
                logits = self.classifier(weighted_out)
            
        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss(weight=self.weight)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs + (sequence_output,)
            
        return outputs  # (loss), logits, (hidden_states), (attentions), sequence_output
        
        
        

        
class TRANSFORMER(classification_model.ClassificationModel):
    def __init__(self, model_type, model_name, num_labels=None, weight=None, sliding_window=False, extract_embeddings=False, args=None, use_cuda=True, classif_type = 'mlp', hidden_dropout_prob=0.1,attention_probs_dropout_prob=0.1):
        super(TRANSFORMER, self).__init__(model_type=model_type, model_name=model_name, num_labels=num_labels, weight=weight, sliding_window=sliding_window, args=args, use_cuda=use_cuda)
        
        
        MODEL_CLASSES = {
            "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
            "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
        }
        
        
        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        if num_labels:
            self.config = config_class.from_pretrained(model_name, num_labels=num_labels, hidden_dropout_prob=hidden_dropout_prob, attention_probs_dropout_prob=attention_probs_dropout_prob)
            self.num_labels = num_labels
        else:
            self.config = config_class.from_pretrained(model_name, hidden_dropout_prob=hidden_dropout_prob, attention_probs_dropout_prob=attention_probs_dropout_prob)
            self.num_labels = self.config.num_labels
        self.tokenizer = tokenizer_class.from_pretrained(model_name)
        self.num_labels = num_labels
        self.weight = weight
        self.sliding_window = sliding_window
        self.extract_embeddings = extract_embeddings
        self.classif_type = classif_type
        
        if use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable. Make sure CUDA is available or set use_cuda=False."
                )
        else:
            self.device = "cpu"

        if self.weight:
            self.model = model_class.from_pretrained(
                model_name,
                config=self.config,
                weight=torch.Tensor(self.weight).to(self.device),
                sliding_window=self.sliding_window,
                classif_type = self.classif_type,
            )
        else:
            self.model = model_class.from_pretrained(model_name, config=self.config, sliding_window=self.sliding_window, extract_embeddings=self.extract_embeddings, classif_type=self.classif_type)
        

        self.results = {}
        
        self.args = {
          "output_dir": "outputs/",
          "cache_dir": "cache/",
        
          "fp16": True,
          "fp16_opt_level": "O1",
          "max_seq_length": 256,
          "train_batch_size": 8,
          "eval_batch_size": 8,
          "gradient_accumulation_steps": 1,
          "num_train_epochs": 1,
          "weight_decay": 0,
          "learning_rate": 4e-5,
          "adam_epsilon": 1e-8,
          "warmup_ratio": 0.06,
          "warmup_steps": 0,
          "max_grad_norm": 1.0,
          "do_lower_case": False,
          "stride": 0.8,
        
          "logging_steps": 50,
          "evaluate_during_training": False,
          "evaluate_during_epoch_ratio": 3,
          "evaluate_during_training_verbose": False,
          "use_cached_eval_features": True,
          "save_eval_checkpoints": True,
          "save_steps": 2000,
          "no_cache": False,
          "save_model_every_epoch": False,
          "tensorboard_dir": None,
        
          "overwrite_output_dir": True,
          "reprocess_input_data": False,
          
          "process_count": cpu_count() - 2 if cpu_count() > 2 else 1,
          "n_gpu": 1,
          "silent": True,
          "use_multiprocessing": False,
        
          "wandb_project": None,
          "wandb_kwargs": {},
        
          "use_early_stopping": True,
          "early_stopping_patience": 3,
          "early_stopping_delta": 0,
        
          "manual_seed": None,
        }
        
        if not use_cuda:
            self.args["fp16"] = False

        if args:
            self.args.update(args)
            
        # if args["n_gpu"] > 1:
        #     self.model = torch.nn.DataParallel(self.model)

        self.args["model_name"] = model_name
        self.args["model_type"] = model_type
        
        # Seeds for reproduceable runs
        torch.manual_seed(args['manual_seed'])
        torch.cuda.manual_seed(args['manual_seed'])
        np.random.seed(args['manual_seed'])
        random.seed(args['manual_seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if model_type == "camembert":
            warnings.warn(
                "use_multiprocessing automatically disabled as CamemBERT fails when using multiprocessing for feature conversion."
            )
            self.args["use_multiprocessing"] = False

        if self.args["stride"] and not sliding_window:
            print("\n[!] WARNING:  Stride argument specified but sliding_window is disabled. Stride will be ignored.\n")
            # warnings.warn("Stride argument specified but sliding_window is disabled. Stride will be ignored.")
        

    def train_model(self, train_df, multi_label=False, output_dir=None, show_running_loss=True, args=None, verbose =True, eval_df=None, writer=None, freeze=False):
        """
        Trains the model using 'train_df'
        Args:
            train_df: Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present,
            the Dataframe should contain at least two columns, with the first column containing the text, and the second column containing the label. The model will be trained on this Dataframe.
            output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            eval_df (optional): A DataFrame against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.
        Returns:
            None
        """

        if args:
            self.args.update(args)

        if self.args["silent"]:
            show_running_loss = False

        if self.args["evaluate_during_training"] and eval_df is None:
            raise ValueError("evaluate_during_training is enabled but eval_df is not specified. Pass eval_df to model.train_model() if using evaluate_during_training.")

        if not output_dir:
            output_dir = self.args["output_dir"]

        if os.path.exists(output_dir) and os.listdir(output_dir) and not self.args["overwrite_output_dir"]:
            raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(output_dir))


        self._move_model_to_device()

        if "text" in train_df.columns and "labels" in train_df.columns:
            train_examples = [InputExample(i, text, None, label) for i, (text, label) in enumerate(zip(train_df["text"], train_df["labels"]))]
        else:
            train_examples = [InputExample(i, text, None, label) for i, (text, label) in enumerate(zip(train_df.iloc[:, 0], train_df.iloc[:, 1]))]
            
        train_dataset = self.load_and_cache_examples(train_examples)
        print("Running Training with freeze =  ", freeze)
        global_step, tr_loss = self.train(train_dataset, output_dir, show_running_loss=show_running_loss, eval_df=eval_df, verbose=verbose, tb_writer=writer, freeze=freeze)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        print("Training of {} model complete. Best model, tokenizer and args are saved to {}.".format(self.args["model_type"], output_dir))
        
        
        

    def train(self, train_dataset, output_dir, show_running_loss=True, eval_df=None, verbose=True, tb_writer=None, freeze = False):
        """
        Trains the model on train_dataset.
        Utility function to be used by the train_model() method. Not intended to be used directly.
        """
        tokenizer = self.tokenizer
        device = self.device
        model = self.model
        args = self.args
        

        # tb_writer = SummaryWriter(logdir=args["tensorboard_dir"])
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args["train_batch_size"])
        print("\nSteps in one epoch  =  ", len(train_dataloader))
        print("Effective Global steps (based on grads accuml. steps)  = ", int(np.floor(len(train_dataloader) / args["gradient_accumulation_steps"])))

        t_total = len(train_dataloader) // args["gradient_accumulation_steps"] * args["num_train_epochs"]
    
        evaluate_during_training_steps = int(np.floor(len(train_dataloader) / (args["gradient_accumulation_steps"]*args['evaluate_during_epoch_ratio'])))
        if args['evaluate_during_training']:
            print("Evalutions done during training every {}  steps\n".format(evaluate_during_training_steps))

        no_decay = ["bias", "LayerNorm.weight"]
        
        # tunable_layers = {str(l) for l in range(8, 12)}
        # for name, param in self.bert.named_parameters():
        #     if not set.intersection(set(name.split('.')), tunable_layers):
        #         param.requires_grad = False
        
        if not freeze:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": args["weight_decay"],
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
        else:
            print("[!]  Model frozen. Just training the classifier\n")
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.classifier.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": args["weight_decay"],
                },
                {
                    "params": [p for n, p in model.classifier.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

        warmup_steps = math.ceil(t_total * args["warmup_ratio"])
        args["warmup_steps"] = warmup_steps if args["warmup_steps"] == 0 else args["warmup_steps"]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args["learning_rate"], eps=args["adam_epsilon"])
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args["warmup_steps"], num_training_steps=t_total)

        if args["fp16"]:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

            model, optimizer = amp.initialize(model, optimizer, opt_level=args["fp16_opt_level"])

        if args["n_gpu"] > 1:
            model = torch.nn.DataParallel(model)

        global_step = 0
        best_eval_loss, best_f1 = None, None
        early_stopping_counter=0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        # train_iterator = trange(int(args["num_train_epochs"]), desc="Epoch", disable=args["silent"])

        model.train()
        for epoch in range(args["num_train_epochs"]):
        # for epoch in range(1):
            # epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(train_dataloader):
                model.train()
                batch = tuple(t.to(self.device) for t in batch)

                inputs = self._get_inputs_dict(batch)
                if self.sliding_window:
                    outputs = model(inputs)
                else:
                    outputs = model(**inputs)
                # model outputs are always tuple in pytorch-transformers (see doc)
                loss = outputs[0]
                if show_running_loss:
                    print("\rRunning loss: %f" % loss, end="")

                if args["n_gpu"] > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args["gradient_accumulation_steps"] > 1:
                    loss = loss / args["gradient_accumulation_steps"]

                if args["fp16"]:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args["max_grad_norm"])
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args["max_grad_norm"])

                tr_loss += loss.item()
                if (step + 1) % args["gradient_accumulation_steps"] == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if args["logging_steps"] > 0 and global_step % args["logging_steps"] == 0:
                        tb_writer.add_scalar("Training/Learning_rate", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar("Training/loss", (tr_loss - logging_loss) / args["logging_steps"], global_step)
                        logging_loss = tr_loss
                    
                    # Log metrics
                    if args["evaluate_during_training"] and (evaluate_during_training_steps > 0 and global_step % evaluate_during_training_steps == 0):
                        # Only evaluate when single GPU otherwise metrics may not average well
                        results, _, _, stats, _, _ = self.eval_model(eval_df, verbose=args["evaluate_during_training_verbose"])
                        for key, value in results.items():
                            tb_writer.add_scalar("Val_conf_matrix/{}".format(key), value, global_step)
                        
                        if self.num_labels == 2:
                            f1_pos, f1_neg, macro_f1, micro_f1, recall, prec, accuracy = stats
                            print("F1 = ", f1_pos)
                            print("macro-f1 = ", macro_f1)
                            
                            tb_writer.add_scalar("Val_results/Precision", prec, global_step)
                            tb_writer.add_scalar("Val_results/Recall", recall, global_step)
                            tb_writer.add_scalar("Val_results/F1_fake", f1_pos, global_step)
                            tb_writer.add_scalar("Val_results/F1_real", f1_neg, global_step)
                            tb_writer.add_scalar("Val_results/macro-F1", macro_f1, global_step)
                            tb_writer.add_scalar("Val_results/micro-F1", micro_f1, global_step)
                            tb_writer.add_scalar("Val_results/Accuracy", accuracy, global_step)
                        
                        elif self.num_labels > 2 :
                            f1, recall, prec, accuracy = stats
                            f1_micro, f1_macro, f1_weighted = f1
                            recall_micro, recall_macro, recall_weighted = recall
                            precision_micro, precision_macro, precision_weighted = prec
                            
                            tb_writer.add_scalar('Validation_F1/macro', macro_f1, epoch)
                            tb_writer.add_scalar('Validation_F1/micro', f1_micro, epoch)
                            tb_writer.add_scalar('Validation_F1/weighted', f1_weighted, epoch)
                            
                            tb_writer.add_scalar('Validation_Precision/macro', precision_macro, epoch)
                            tb_writer.add_scalar('Validation_Precision/micro', precision_micro, epoch)
                            tb_writer.add_scalar('Validation_Precision/weighted', precision_weighted, epoch)            
                            
                            tb_writer.add_scalar('Validation_Recall/macro', recall_macro, epoch)
                            tb_writer.add_scalar('Validation_Recall/micro', recall_micro, epoch)
                            tb_writer.add_scalar('Validation_Recall/weighted', recall_weighted, epoch)
                        
                        # if not best_eval_loss:
                        #     best_eval_loss = results["eval_loss"]
                        #     self._save_model(args["output_dir"], model=model, results=results)
                        # elif results["eval_loss"] < best_eval_loss:
                        #     best_eval_loss = results["eval_loss"]
                        #     self._save_model(args["output_dir"], model=model, results=results)
                        #     early_stopping_counter = 0
                        # else:
                        #     if args["use_early_stopping"]:
                        #         if early_stopping_counter < args["early_stopping_patience"]:
                        #             early_stopping_counter += 1
                        #             if verbose:
                        #                 print("No improvement in eval_loss for {} steps.\n".format(early_stopping_counter))
                        #         else:
                        #             if verbose:
                        #                 print("\nPatience of {} steps reached.\nTraining terminated.".format(args['early_stopping_patience']))
                        #             return global_step, tr_loss / global_step
                            
                            
                        if not best_f1:
                            best_f1 = macro_f1
                            self._save_model(args["output_dir"], model=model, results=results)
                        elif macro_f1 > best_f1:
                            best_f1 = macro_f1
                            print("best F1 = ", best_f1)
                            self._save_model(args["output_dir"], model=model, results=results)
                            early_stopping_counter = 0
                        else:
                            if args["use_early_stopping"]:
                                if early_stopping_counter < args["early_stopping_patience"]:
                                    early_stopping_counter += 1
                                    if verbose:
                                        print("No improvement in eval_loss for {} steps.\n".format(early_stopping_counter))
                                else:
                                    if verbose:
                                        print("\nPatience of {} steps reached.\nTraining terminated.".format(args['early_stopping_patience']))
                                    return global_step, tr_loss / global_step

                    if args["save_steps"] > 0 and global_step % args["save_steps"] == 0:
                        # Save model checkpoint
                        output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                        os.makedirs(output_dir_current, exist_ok=True)

                        # Take care of distributed/parallel training
                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrained(output_dir_current)
                        self.tokenizer.save_pretrained(output_dir_current)
            # self._save_model(args["output_dir"], model=model, results=None)
            print("End of epoch {}\n".format(epoch+1))        
            if not args["evaluate_during_training"]:
                # Only evaluate when single GPU otherwise metrics may not average well
                results, _, _, stats, _, _ = self.eval_model(eval_df, verbose=args["evaluate_during_training_verbose"])
                for key, value in results.items():
                    tb_writer.add_scalar("Val_conf_matrix/{}".format(key), value, global_step)
                
                if self.num_labels == 2:
                    f1_pos, f1_neg, macro_f1, micro_f1, recall, prec, accuracy = stats
                    
                    tb_writer.add_scalar("Val_results/Precision", prec, global_step)
                    tb_writer.add_scalar("Val_results/Recall", recall, global_step)
                    tb_writer.add_scalar("Val_results/F1_fake", f1_pos, global_step)
                    tb_writer.add_scalar("Val_results/F1_real", f1_neg, global_step)
                    tb_writer.add_scalar("Val_results/macro-F1", macro_f1, global_step)
                    tb_writer.add_scalar("Val_results/micro-F1", micro_f1, global_step)
                    tb_writer.add_scalar("Val_results/Accuracy", accuracy, global_step)
                
                elif self.num_labels >2 :
                    f1, recall, prec, accuracy = stats
                    f1_micro, f1_macro, f1_weighted = f1
                    recall_micro, recall_macro, recall_weighted = recall
                    precision_micro, precision_macro, precision_weighted = prec
                    
                    tb_writer.add_scalar('Validation_F1/macro', macro_f1, epoch)
                    tb_writer.add_scalar('Validation_F1/micro', f1_micro, epoch)
                    tb_writer.add_scalar('Validation_F1/weighted', f1_weighted, epoch)
                    
                    tb_writer.add_scalar('Validation_Precision/macro', precision_macro, epoch)
                    tb_writer.add_scalar('Validation_Precision/micro', precision_micro, epoch)
                    tb_writer.add_scalar('Validation_Precision/weighted', precision_weighted, epoch)            
                    
                    tb_writer.add_scalar('Validation_Recall/macro', recall_macro, epoch)
                    tb_writer.add_scalar('Validation_Recall/micro', recall_micro, epoch)
                    tb_writer.add_scalar('Validation_Recall/weighted', recall_weighted, epoch)
                    
                    
                
                # if not best_eval_loss:
                #     best_eval_loss = results["eval_loss"]
                #     self._save_model(args["output_dir"], model=model, results=results)
                # elif results["eval_loss"] < best_eval_loss:
                #     best_eval_loss = results["eval_loss"]
                #     self._save_model(args["output_dir"], model=model, results=results)
                #     early_stopping_counter = 0
                # else:
                #     if args["use_early_stopping"]:
                #         if early_stopping_counter < args["early_stopping_patience"]:
                #             early_stopping_counter += 1
                #             if verbose:
                #                 print("No improvement in eval_loss for {} steps.\n".format(early_stopping_counter))
                #         else:
                #             if verbose:
                #                 print("\nPatience of {} steps reached.\nTraining terminated.".format(args['early_stopping_patience']))
                #             return global_step, tr_loss / global_step
                    
                    
                if not best_f1:
                    best_f1 = macro_f1
                    self._save_model(args["output_dir"], model=model, results=results)
                elif macro_f1 > best_f1:
                    best_f1 = macro_f1
                    print("best F1 = ", best_f1)
                    self._save_model(args["output_dir"], model=model, results=results)
                    early_stopping_counter = 0
                else:
                    if args["use_early_stopping"]:
                        if early_stopping_counter < args["early_stopping_patience"]:
                            early_stopping_counter += 1
                            if verbose:
                                print("No improvement in eval_loss for {} steps.\n".format(early_stopping_counter))
                        else:
                            if verbose:
                                print("\nPatience of {} steps reached.\nTraining terminated.".format(args['early_stopping_patience']))
                            return global_step, tr_loss / global_step

        return global_step, tr_loss / global_step
    
    
    

    def eval_model(self, eval_df, multi_label=False, output_dir=None, verbose=False, test=False, **kwargs):
        """
        Evaluates the model on eval_df. Saves results to output_dir.
        Args:
            eval_df: Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present,
            the Dataframe should contain at least two columns, with the first column containing the text, and the second column containing the label. The model will be evaluated on this Dataframe.
            output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.
            verbose: If verbose, results will be printed to the console on completion of evaluation.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.
        Returns:
            result: Dictionary containing evaluation results. (Matthews correlation coefficient, tp, tn, fp, fn)
            model_outputs: List of model outputs for each row in eval_df
            wrong_preds: List of InputExample objects corresponding to each incorrect prediction by the model
        """

        if not output_dir:
            output_dir = self.args["output_dir"]

        self._move_model_to_device()

        result, model_outputs, wrong_preds, stats, embeds, labels = self.evaluate(eval_df, output_dir, multi_label=multi_label, test=test, **kwargs)
        self.results.update(result)

        # print("Embeds = ", embeds.shape)
        if verbose:
            print(self.results)
        return result, model_outputs, wrong_preds, stats, embeds, labels
    
    
    

    def evaluate(self, eval_df, output_dir, multi_label=False, test=False, prefix="", **kwargs):
        """
        Evaluates the model on eval_df.
        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """

        tokenizer = self.tokenizer
        device = self.device
        model = self.model
        args = self.args
        eval_output_dir = output_dir
        
        if not test:
            model = torch.nn.DataParallel(model)

        results = {}

        if "text" in eval_df.columns and "labels" in eval_df.columns:
            eval_examples = [
                InputExample(i, text, None, label)
                for i, (text, label) in enumerate(zip(eval_df["text"], eval_df["labels"]))
            ]
        else:
            eval_examples = [
                InputExample(i, text, None, label)
                for i, (text, label) in enumerate(zip(eval_df.iloc[:, 0], eval_df.iloc[:, 1]))
            ]

        eval_dataset = self.load_and_cache_examples(eval_examples, evaluate=True)
        os.makedirs(eval_output_dir, exist_ok=True)

        eval_sampler = SequentialSampler(eval_dataset)
        batch = args["eval_batch_size"] if not test else 2
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch)

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        embeds= None
        model.eval()

        for batch in tqdm(eval_dataloader, disable=args["silent"]):
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = self._get_inputs_dict(batch)

                if self.sliding_window:
                    outputs = model(inputs)
                else:
                    outputs = model(**inputs)

                tmp_eval_loss, logits = outputs[:2]
                if embeds is None:
                    temp = outputs[-1]
                    embeds = temp[: ,0, :]
                else:
                    temp = outputs[-1]
                    embeds = torch.cat((embeds, temp[: ,0, :]), 0)

                if multi_label:
                    logits = logits.sigmoid()
                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1

            if self.sliding_window:
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs[0]["labels"].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, inputs[0]["labels"].detach().cpu().numpy(), axis=0)
            else:
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs["labels"].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        model_outputs = preds

        if not multi_label:
            preds = np.argmax(preds, axis=1)

        result, wrong, stats= self.compute_metrics(preds, out_label_ids, eval_examples, **kwargs)
        result["eval_loss"] = eval_loss
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))
        return results, model_outputs, wrong, stats, embeds, torch.tensor(out_label_ids)
    
    
    

    def load_and_cache_examples(self, examples, evaluate=False, no_cache=False, multi_label=False):
        """
        Converts a list of InputExample objects to a TensorDataset containing InputFeatures. Caches the InputFeatures.
        Utility function for train() and eval() methods. Not intended to be used directly.
        """

        process_count = self.args["process_count"]

        tokenizer = self.tokenizer
        output_mode = "classification"
        args = self.args

        if not os.path.isdir(self.args["cache_dir"]):
            os.makedirs(self.args["cache_dir"])

        mode = "dev" if evaluate else "train"
        cached_features_file = os.path.join(
            args["cache_dir"],
            "cached_{}_{}_{}_{}_{}".format(
                mode, args["model_type"], args["max_seq_length"], self.num_labels, len(examples)
            ),
        )

        if os.path.exists(cached_features_file) and not args["reprocess_input_data"] and not no_cache:
            features = torch.load(cached_features_file)
            print(f"Features loaded from cache at {cached_features_file}")
        else:
            # print(f"Converting to features started.")
            # print(examples)
            features = convert_examples_to_features(
                examples,
                args["max_seq_length"],
                tokenizer,
                output_mode,
                # XLNet has a CLS token at the end
                cls_token_at_end=bool(args["model_type"] in ["xlnet"]),
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if args["model_type"] in ["xlnet"] else 0,
                sep_token=tokenizer.sep_token,
                # RoBERTa uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                sep_token_extra=bool(args["model_type"] in ["roberta"]),
                # PAD on the left for XLNet
                pad_on_left=bool(args["model_type"] in ["xlnet"]),
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args["model_type"] in ["xlnet"] else 0,
                process_count=process_count,
                multi_label=multi_label,
                silent=args["silent"],
                use_multiprocessing=args["use_multiprocessing"],
                sliding_window=self.sliding_window,
                stride=self.args["stride"],
            )

            if not no_cache:
                torch.save(features, cached_features_file)

        if self.sliding_window:
            # features = pad_sequence([torch.tensor(features_per_sequence) for features_per_sequence in features])
            all_input_ids = pad_sequence(
                [
                    torch.tensor([f.input_ids for f in features_per_sequence], dtype=torch.long)
                    for features_per_sequence in features
                ],
                batch_first=True,
            )
            all_input_mask = pad_sequence(
                [
                    torch.tensor([f.input_mask for f in features_per_sequence], dtype=torch.long)
                    for features_per_sequence in features
                ],
                batch_first=True,
            )
            all_segment_ids = pad_sequence(
                [
                    torch.tensor([f.segment_ids for f in features_per_sequence], dtype=torch.long)
                    for features_per_sequence in features
                ],
                batch_first=True,
            )


            if output_mode == "classification":
                all_label_ids = pad_sequence(
                    [
                        torch.tensor([f.label_id for f in features_per_sequence], dtype=torch.long)
                        for features_per_sequence in features
                    ],
                    batch_first=True,
                )
            elif output_mode == "regression":
                all_label_ids = pad_sequence(
                    [
                        torch.tensor([f.label_id for f in features_per_sequence], dtype=torch.float)
                        for features_per_sequence in features
                    ],
                    batch_first=True,
                )
        else:
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

            if output_mode == "classification":
                all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
            elif output_mode == "regression":
                all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        return dataset
    
    

    def compute_metrics(self, preds, labels, eval_examples, multi_label=False, **kwargs):
        """
        Computes the evaluation metrics for the model predictions.
        Args:
            preds: Model predictions
            labels: Ground truth labels
            eval_examples: List of examples on which evaluation was performed
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.
        Returns:
            result: Dictionary containing evaluation results. (Matthews correlation coefficient, tp, tn, fp, fn)
            wrong: List of InputExample objects corresponding to each incorrect prediction by the model
        """

        assert len(preds) == len(labels)

        extra_metrics = {}
        for metric, func in kwargs.items():
            extra_metrics[metric] = func(labels, preds)

        mismatched = labels != preds
        wrong = [i for (i, v) in zip(eval_examples, mismatched) if v.any()]

        if multi_label:
            label_ranking_score = label_ranking_average_precision_score(labels, preds)
            return {**{"LRAP": label_ranking_score}, **extra_metrics}, wrong

        mcc = matthews_corrcoef(labels, preds)

        if self.model.num_labels ==2:
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            f1_pos = f1_score(labels, preds, average = 'binary', pos_label =1)
            f1_neg = f1_score(labels, preds, average = 'binary', pos_label =0)
            macro_f1 = f1_score(labels, preds, average = 'macro', pos_label=1)
            micro_f1 = f1_score(labels, preds, average = 'micro', pos_label=1)
            recall = recall_score(labels, preds, average = 'binary', pos_label=1)
            precision = precision_score(labels, preds, average = 'binary', pos_label=1)
            accuracy = accuracy_score(labels, preds)
            return {**{"mcc": mcc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}, **extra_metrics}, wrong, (f1_pos, f1_neg, macro_f1, micro_f1, recall, precision, accuracy)
        
        elif self.model.num_labels >2:
            f1_micro = f1_score(labels, preds, average = 'micro')
            f1_macro = f1_score(labels, preds, average = 'macro')
            f1_weighted = f1_score(labels, preds, average = 'weighted')
            
            recall_macro = recall_score(labels, preds, average = 'macro')
            recall_micro = recall_score(labels, preds, average = 'micro')
            recall_weighted = recall_score(labels, preds, average = 'weighted')
            
            precision_micro = precision_score(labels, preds, average = 'micro')
            precision_macro = precision_score(labels, preds, average = 'macro')
            precision_weighted = precision_score(labels, preds, average = 'weighted')
            accuracy = accuracy_score(labels, preds)
            
            f1 = f1_micro, f1_macro, f1_weighted
            recall = recall_micro, recall_macro, recall_weighted
            prec = precision_micro, precision_macro, precision_weighted
            
            return {**{"mcc": mcc}, **extra_metrics}, wrong, (f1, recall, prec, accuracy)
            
        elif self.model.num_labels == 1:
            return {**{"mcc": mcc}, **extra_metrics}, wrong

    def predict(self, to_predict, multi_label=False):
        """
        Performs predictions on a list of text.
        Args:
            to_predict: A python list of text (str) to be sent to the model for prediction.
        Returns:
            preds: A python list of the predictions (0 or 1) for each text.
            model_outputs: A python list of the raw model outputs for each text.
        """

        tokenizer = self.tokenizer
        device = self.device
        model = self.model
        args = self.args

        self._move_model_to_device()

        if multi_label:
            eval_examples = [
                InputExample(i, text, None, [0 for i in range(self.num_labels)]) for i, text in enumerate(to_predict)
            ]
        else:
            eval_examples = [InputExample(i, text, None, 0) for i, text in enumerate(to_predict)]

        eval_dataset = self.load_and_cache_examples(
            eval_examples, evaluate=True, multi_label=multi_label, no_cache=True
        )

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args["eval_batch_size"])

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, disable=args["silent"]):
            model.eval()
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = self._get_inputs_dict(batch)

                if self.sliding_window:
                    outputs = model(inputs)
                else:
                    outputs = model(**inputs)

                tmp_eval_loss, logits = outputs[:2]

                if multi_label:
                    logits = logits.sigmoid()

                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1

            if self.sliding_window:
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs[0]["labels"].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, inputs[0]["labels"].detach().cpu().numpy(), axis=0)
            else:
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs["labels"].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        model_outputs = preds
        if multi_label:
            if isinstance(args["threshold"], list):
                threshold_values = args["threshold"]
                preds = [
                    [self._threshold(pred, threshold_values[i]) for i, pred in enumerate(example)] for example in preds
                ]
            else:
                preds = [[self._threshold(pred, args["threshold"]) for pred in example] for example in preds]
        else:
            preds = np.argmax(preds, axis=1)
        
        doc_embed = outputs[-1]
        return preds, model_outputs, doc_embed

    def _threshold(self, x, threshold):
        if x >= threshold:
            return 1
        return 0

    def _move_model_to_device(self):
        self.model.to(self.device)

    def _get_inputs_dict(self, batch):
        if self.sliding_window:
            inputs_all = []
            inputs = batch[0].permute(1, 0, 2)
            attentions = batch[1].permute(1, 0, 2)
            labels = batch[3].permute(1, 0)

            if self.args["model_type"] != "distilbert":
                tokens = batch[2].permute(1, 0, 2) if self.args["model_type"] in ["bert", "xlnet"] else None

            for i in range(len(labels)):
                input_single = {"input_ids": inputs[i], "attention_mask": attentions[i], "labels": labels[i]}

                # XLM, DistilBERT and RoBERTa don't use segment_ids
                if self.args["model_type"] != "distilbert":
                    input_single["token_type_ids"] = tokens[i] if self.args["model_type"] in ["bert", "xlnet"] else None
                inputs_all.append(input_single)
            return inputs_all
        else:
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

            # XLM, DistilBERT and RoBERTa don't use segment_ids
            if self.args["model_type"] != "distilbert":
                inputs["token_type_ids"] = batch[2] if self.args["model_type"] in ["bert", "xlnet"] else None

            return inputs
        
        
    def _save_model(self, output_dir, model=None, results=None):
        os.makedirs(output_dir, exist_ok=True)

        if model:
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            print("Saving model and tokenizer!!")
        
        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))