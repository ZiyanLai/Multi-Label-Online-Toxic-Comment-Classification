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
# pretrained_weights = "bert-base-uncased"
pretrained_weights = "bert_model"
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




class LSTMClassifier(nn.Module):
    """
    BERTClassifier classification model
    """
    def __init__(self, bert, hidden_size, num_classes, num_layers, bidirectional, dropout_prob=0.3):
        if bidirectional == True:
            fac = 2
        else:
            fac = 1
        super().__init__()
        self.bert = bert.to(device)
        self.non_linearity = nn.ReLU().to(device)
        self.clf = nn.Linear(hidden_size*fac, num_classes).to(device)
#         self.linear_layer = nn.Linear(bert.config.hidden_size, hidden_size).to(device)
        self.dropout = nn.Dropout(dropout_prob).to(device)
        self.lstm = nn.LSTM(bert.config.hidden_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True, dropout = dropout_prob).to(device)
        self.pool = nn.MaxPool1d(1).to(device)


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
#         print(input_ids)
        embedded = outputs[0].to(device)
#         print(embedded.shape)
        lstm, _ = self.lstm(embedded)
        lstm = lstm.to(device)
#         print(lstm.shape)
        num_embedded = (input_ids==0).float().sum(1).clamp(1)
#         print(num_embedded)
        pooled = self.pool(lstm).to(device)


        average_embedding = pooled.sum(1) / num_embedded.view(-1,1) 
        
#         embedded = self.dropout(embedded).to(device)

#         linear = self.linear_layer(embedded).to(device)
#         print(average_embedding)
        linear = self.clf(average_embedding).to(device)
        
        logits = self.non_linearity(linear).to(device)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels.type(torch.LongTensor).to(device))
        # logits = F.softmax(non_linear, dim=1).to(device)
        # print(logits)
        return logits, loss



def evaluate(model, dataloader):
    model.eval()
    """
        4. TODO: Your code here
        Calculate the accuracy of the model on the data in dataloader
        You may refer to `run_inference` function from Lab2 
    """
    with torch.no_grad():
        all_preds = []
        all_labels = []

        for batch_ind, batch_labels in dataloader:
           
            preds_batch, loss = model(batch_ind, labels = batch_labels)

            for preds in preds_batch:
                # print(preds)
                preds = [preds.tolist()]
                all_preds.append(preds)
            batch_labels = batch_labels.type(torch.LongTensor).tolist()
            # batch_labels = batch_labels.tolist()
            all_labels += batch_labels
        all_preds = np.concatenate(all_preds, axis=0)
        preds = np.argmax(all_preds, axis = -1)
        # print(preds)
        all_labels = np.array(all_labels)
        # print(all_labels)
        acc = accuracy_score(all_labels, preds)
        f1 = f1_score(all_labels, preds)
    return acc, f1




NUM_EPOCHS=5

train_list = [train_threat_loader, train_insult_loader, train_toxic_loader, train_isAbuse_loader]
val_list = [val_threat_loader, val_insult_loader, val_toxic_loader, val_isAbuse_loader]
name_list = ["threat", "insult", "toxic", "IsAbuse"]

all_acc = []
all_f1 = []
all_loss = []

for i in tqdm.tqdm(range(len(train_list))):
    num_labels = 2
    hidden_size = 64
    num_layers = 1
    bidirectional = True
    model = LSTMClassifier(bert, hidden_size, num_labels, num_layers, bidirectional)
    model.to(device)
    optimizer = BertAdam(model.parameters(), lr=2e-6)

    train_loss_history = []
    val_accuracy_history = []
    val_f1_history = []

    train_loader = train_list[i]
    val_loader = val_list[i]

    for epoch in tqdm.tqdm(range(NUM_EPOCHS)):
        model.train() # this enables regularization, which we don't currently have
        for j, (data_batch, batch_labels) in enumerate(train_loader):
            """
                Code for training lstm
                Keep track of training of for each batch using train_loss_history
            """
    #        
            preds, loss = model(data_batch, labels = batch_labels)
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

        accuracy, f1 = evaluate(model, val_loader)
            
    #         print("current accuracy: ", accuracy)
    #         print("current f1: ", f1)
            
        val_accuracy_history.append(accuracy)
        val_f1_history.append(f1)

        if len(val_f1_history) == 1:
                curr_max = val_f1_history[0]
                torch.save(model, 'LSTM_best_model-'+pretrained_weights+'-'+name_list[i]+'.pt')
        elif len(val_f1_history) > 1:
            if f1 > curr_max:
                curr_max = f1
                torch.save(model, 'LSTM_best_model-'+pretrained_weights+'-'+name_list[i]+'.pt')

    all_acc.append(val_accuracy_history)
    all_f1.append(val_f1_history)
    all_loss.append(train_loss_history)

for i in range(len(train_list)):
    print("Current Label: ", name_list[i])
    print("acc: ", all_acc[i])
    print("F1: ", all_f1[i])
    print("Loss History: ", all_loss[i][-50:])
    print('\n')
