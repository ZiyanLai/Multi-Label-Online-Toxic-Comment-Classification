#!/usr/bin/env python
# coding: utf-8


import transformers
from transformers import BertModel, BertTokenizer
from pytorch_pretrained_bert import BertAdam
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score





if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("I am using gpu", device)
else:
    device = torch.device('cpu')



# tokenizer = BertTokenizer.from_pretrained('bert_model')
# bert = BertModel.from_pretrained('bert_model')
pretrained_weights = "bert_model"
# pretrained_weights = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
bert = BertModel.from_pretrained(pretrained_weights)




train = pd.read_csv('clean_train.csv')
val = pd.read_csv('clean_val.csv')
test = pd.read_csv('clean_test.csv')

np.random.seed(42)
train = train.sample(frac=1)
val = val.sample(frac=1)
test = test.sample(frac=1)
train = train.reset_index(drop=True)
val = val.reset_index(drop=True)
test=test.reset_index(drop=True)

train_examples = 30000
test_examples = 3500
val_examples = 3500
# train_examples = 3000
# test_examples = 350
# val_examples = 350

train = train.iloc[:train_examples,:]
val = val.iloc[:test_examples,:]
test = test.iloc[:val_examples,:]




def featurize(data, labels):
    text_ids = []
    text_attentions = []
    for ind, sentence in tqdm.tqdm(data.items()):
#         
        input_dic = tokenizer.encode_plus(sentence, max_length=128, add_special_tokens=True, pad_to_max_length=True, return_tensors = 'pt')
        input_ids = input_dic['input_ids']
        input_attention = input_dic['attention_mask']
#         
        text_ids.append(input_ids)
        text_attentions.append(input_attention)
        labels = torch.FloatTensor(labels)
    return text_ids, text_attentions, labels.to(device)



def create_sampler(label):
    len_positive_class = len(train[train[label] == 1])
    len_negative_class = len(train[train[label] == 0])
    # print(len_negative_class)
    # print(len_positive_class)
    class_sample = torch.Tensor([len_negative_class, len_positive_class])
    class_weights = 1./class_sample
    # print(class_weights)
    samples_weight = [class_weights[i].tolist() for i in train[label]]
    # print(samples_weight)
    # for t in train['IsAbuse']:
    #     print(class_weights[t])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler

isAbuse_sampler = create_sampler('IsAbuse')
insult_sampler = create_sampler('insult')
threat_sampler = create_sampler('threat')
toxic_sampler = create_sampler('toxic')


train_data_label = train[['threat','insult','toxic','IsAbuse']].values.tolist()
train_data_ind, train_attention, train_data_label = featurize(train.comment_text, train_data_label)

val_data_label = val[['threat','insult','toxic','IsAbuse']].values.tolist()
val_data_ind, val_attention, val_data_label = featurize(val.comment_text, val_data_label)

test_data_label = test[['threat','insult','toxic','IsAbuse']].values.tolist()
test_data_ind, test_attention, test_data_label = featurize(test.comment_text, test_data_label)

train_data_ind = torch.cat(train_data_ind, dim=0).to(device)
# train_attention = torch.cat(train_attention, dim = 0)

val_data_ind = torch.cat(val_data_ind, dim=0).to(device)
# val_attention = torch.cat(val_attention, dim = 0)

test_data_ind = torch.cat(test_data_ind, dim=0).to(device)

train_threat_label = train_data_label[:,0]
val_threat_label = val_data_label[:,0]
test_threat_label = test_data_label[:,0]

train_insult_label = train_data_label[:,1]
val_insult_label = val_data_label[:,1]
test_insult_label = test_data_label[:,1]

train_toxic_label = train_data_label[:,2]
val_toxic_label = val_data_label[:,2]
test_toxic_label = test_data_label[:,2]

train_isAbuse_label = train_data_label[:,3]
val_isAbuse_label = val_data_label[:,3]
test_isAbuse_label = test_data_label[:,3]




# Create Dataset
train_threat_dataset = TensorDataset(train_data_ind, train_threat_label)
val_threat_dataset = TensorDataset(val_data_ind, val_threat_label)
test_threat_dataset = TensorDataset(test_data_ind, test_threat_label)

train_insult_dataset = TensorDataset(train_data_ind, train_insult_label)
val_insult_dataset = TensorDataset(val_data_ind, val_insult_label)
test_insult_dataset = TensorDataset(test_data_ind, test_insult_label)

train_toxic_dataset = TensorDataset(train_data_ind, train_toxic_label)
val_toxic_dataset = TensorDataset(val_data_ind, val_toxic_label)
test_toxic_dataset = TensorDataset(test_data_ind, test_toxic_label)

train_isAbuse_dataset = TensorDataset(train_data_ind, train_isAbuse_label)
val_isAbuse_dataset = TensorDataset(val_data_ind, val_isAbuse_label)
test_isAbuse_dataset = TensorDataset(test_data_ind, test_isAbuse_label)

# Load in to DataLoader
BATCH_SIZE = 10
# Threat
train_threat_loader = DataLoader(train_threat_dataset, batch_size = BATCH_SIZE, shuffle=False, sampler=threat_sampler)
val_threat_loader = DataLoader(val_threat_dataset, batch_size = BATCH_SIZE, shuffle=True)
test_threat_loader = DataLoader(test_threat_dataset, batch_size = BATCH_SIZE, shuffle=True)

# Insult
train_insult_loader = DataLoader(train_insult_dataset, batch_size = BATCH_SIZE, shuffle=False, sampler=insult_sampler)
val_insult_loader = DataLoader(val_insult_dataset, batch_size = BATCH_SIZE, shuffle=True)
test_insult_loader = DataLoader(test_insult_dataset, batch_size = BATCH_SIZE, shuffle=True)

# Toxic
train_toxic_loader = DataLoader(train_toxic_dataset, batch_size = BATCH_SIZE, shuffle=False, sampler=toxic_sampler)
val_toxic_loader = DataLoader(val_toxic_dataset, batch_size = BATCH_SIZE, shuffle=True)
test_toxic_loader = DataLoader(test_toxic_dataset, batch_size = BATCH_SIZE, shuffle=True)

# IsAbuse
train_isAbuse_loader = DataLoader(train_isAbuse_dataset, batch_size = BATCH_SIZE, shuffle=False, sampler=isAbuse_sampler)
val_isAbuse_loader = DataLoader(val_isAbuse_dataset, batch_size = BATCH_SIZE, shuffle=True)
test_isAbuse_loader = DataLoader(test_isAbuse_dataset, batch_size = BATCH_SIZE, shuffle=True)



# i = next(iter(test_isAbuse_loader))
# i




class BERTClassifier(nn.Module):
    """
    BERTClassifier classification model
    """
    def __init__(self, bert, hidden_size, num_classes, dropout_prob=0.3):
        super().__init__()
        self.bert = bert.to(device)
        self.non_linearity = nn.ReLU().to(device)
        self.linear_layer = nn.Linear(bert.config.hidden_size, hidden_size).to(device)
        self.clf = nn.Linear(hidden_size, num_classes).to(device)
        self.sigmoid = nn.Sigmoid().to(device)
        self.dropout = nn.Dropout(dropout_prob).to(device)
        # self.sigmoid = nn.Sigmoid().to(device)
#         self.pool = nn.MaxPool1d(1)
#         self.embedding_layer = self.load_pretrained_embeddings(embeddings)
#         self.lstm = nn.LSTM(EMBEDDING_DIM, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True, dropout = dropout_prob)
#         self.dropout = nn.Dropout(dropout_prob)

#     def load_pretrained_embeddings(self, embeddings):
#         """
#            The code for loading embeddings from Lab 2
#            Unlike lab, we are not setting `embedding_layer.weight.requires_grad = False`
#            because we want to finetune the embeddings on our data
#         """
#         embedding_layer = nn.Embedding(30522, 768, padding_idx=0)
#         embedding_layer.weight.data = torch.Tensor().float()
#         return embedding_layer

    def forward(self, input_ids=None,
                      attention_mask=None,
                      token_type_ids=None,
                      position_ids=None,
                      head_mask=None,
                      inputs_embeds=None,
                      labels=None,):
        
        outputs = self.bert(input_ids, 
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,)
        
        
        embedded = outputs[1]
#         print(embedded)
        # embedded = self.dropout(embedded).to(device)

        linear = self.linear_layer(embedded)
        linear = self.sigmoid(linear)

        # non_linear = self.non_linearity(linear)
        
        logits = self.clf(linear)
        
        logits = self.sigmoid(logits)

        criterion = nn.BCELoss()
        loss = 0
        if labels is not None:
            # print("logits:", logits)
            # print("labels:",labels)
            labels = labels.view(-1,1)
            # print(logits)
            # print(labels)
            loss = criterion(logits, labels)
            # print("loss:", loss)
    
        return loss, logits


def evaluate(model, dataloader, threshold):
    model.eval()
    """
        4. TODO: Your code here
        Calculate the accuracy of the model on the data in dataloader
        You may refer to `run_inference` function from Lab2 
    """
    with torch.no_grad():
        all_preds = []
        all_labels = []
        all_preds_prob = []
        for batch_ind, batch_labels in dataloader:
            # print(batch_labels)
            # labels += batch_labels.cpu().tolist()
            loss, preds_batch = model(batch_ind, labels = batch_labels)

            # print(preds_batch)
#             print(preds_batch)
#             print("batch:", preds_batch)
            for preds in preds_batch:
                # print(preds)
                
                all_preds_prob.append(float(preds))
                if preds >= threshold:
                    all_preds.append(int(1))
                else:
                    all_preds.append(int(0))
                    
#             print(preds_batch)
            
            batch_labels = batch_labels.tolist()
            all_labels += batch_labels

        # all_labels = np.vstack(all_labels)
        # all_preds = np.vstack(all_preds)
        # print(all_preds_prob)
        # print(all_preds)
        # print(all_labels)
        # print(all_preds)

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        roc_auc = roc_auc_score(all_labels, all_preds_prob)
        
    return acc, f1, roc_auc







NUM_EPOCHS = 5
# NUM_EPOCHS =3

threshold = 0.48

train_list = [train_threat_loader, train_insult_loader, train_toxic_loader, train_isAbuse_loader]
val_list = [val_threat_loader, val_insult_loader, val_toxic_loader, val_isAbuse_loader]
name_list = ["threat", "insult", "toxic", "IsAbuse"]
# train_list = [  train_toxic_loader]
# val_list = [  val_toxic_loader]
# name_list = ["toxic"]
all_acc = []
all_f1 = []
all_roc_auc = []
all_loss = []

for i in tqdm.tqdm(range(len(train_list))):

    num_classes = 1
    hidden_size = 64
    model = BERTClassifier(bert, hidden_size, num_classes).to(device)
    torch.manual_seed(1234)
    # criterion = nn.CrossEntropyLoss()
    optimizer = BertAdam(model.parameters(), lr=2e-6)

    train_loss_history = []
    val_accuracy_history = []
    val_f1_history = []
    val_roc_auc_history = []
    # best_f1_accuracy = 0
    train_loader = train_list[i]
    val_loader = val_list[i]


    for epoch in tqdm.tqdm(range(NUM_EPOCHS)):
            model.train() # this enables regularization, which we don't currently have
            for j, (data_batch, batch_labels) in enumerate(train_loader):
                """
                Code for training lstm
                Keep track of training of for each batch using train_loss_history
                """
    #             print(data_batch)
    #             print(batch_labels)
                # data_batch = data_batch.to(device)
                # batch_labels = batch_labels.type(torch.FloatTensor)
                # batch_labels = batch_labels.to(device)
                loss, preds = model(data_batch, labels = batch_labels)
                
    #             print(preds)
                # preds = preds.to(device)
    #             print(preds)
                # loss = criterion(preds, batch_labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss_history.append(loss.item())

            # The end of a training epoch 

            """
                Code for tracking best validation accuracy, saving the best model, and early stopping
                # Compute validation accuracy after each training epoch using `evaluate` function
                # Keep track of validation accuracy in `val_accuracy_history`
                # save model with best validation accuracy, hint: torch.save(model, 'best_model.pt')
                # Early stopping: 
                # stop training if the validation accuracy does not improve for more than `early_stop_patience` runs
                5. TODO: Your code here """

            accuracy, f1, roc_auc = evaluate(model, val_loader, threshold)
            
    #         print("current accuracy: ", accuracy)
    #         print("current f1: ", f1)
            
            val_accuracy_history.append(accuracy)
            val_f1_history.append(f1)
            val_roc_auc_history.append(roc_auc)
            
            if len(val_roc_auc_history) == 1:
                curr_max = val_roc_auc_history[0]
                torch.save(model, 'best_model-'+pretrained_weights+'-'+name_list[i]+'.pt')
            elif len(val_roc_auc_history) > 1:
                if roc_auc > curr_max:
                    curr_max = roc_auc
                    torch.save(model, 'best_model-'+pretrained_weights+'-'+name_list[i]+'.pt')
                    
            # best_f1_accuracy = curr_max
    all_acc.append(val_accuracy_history)
    all_f1.append(val_f1_history)
    all_roc_auc.append(val_roc_auc_history)
    all_loss.append(train_loss_history)

    # print(train_loss_history)
    # print("acc:", val_accuracy_history)
    # print("f1:", val_f1_history)
    # print("Best validation accuracy is: ", best_f1_accuracy)


for i in range(len(train_list)):
    print("Current Label: ", name_list[i])
    print("acc: ", all_acc[i])
    print("F1: ", all_f1[i])
    print("ROC AUC: ", all_roc_auc[i])
    print("Loss History: ", all_loss[i][-50:])
    print('\n')



