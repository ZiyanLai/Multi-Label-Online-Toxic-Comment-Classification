#!/usr/bin/env python
# coding: utf-8


import transformers
from transformers import BertModel, BertTokenizer
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("I am using gpu", device)
else:
    device = torch.device('cpu')



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')




train = pd.read_csv('clean_train.csv')
val = pd.read_csv('clean_val.csv')
test = pd.read_csv('clean_test.csv')

train = train.iloc[:35000]
val = val.iloc[:4500]
test = test.iloc[:4500]



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



train_dataset = TensorDataset(train_data_ind, train_data_label)
val_dataset = TensorDataset(val_data_ind, val_data_label)
test_dataset = TensorDataset(test_data_ind, test_data_label)

BATCH_SIZE = 20
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=True)



# i = next(iter(train_loader))
# i = i[0]
# b = bert(i)[0]



# b.shape


class LSTMClassifier(nn.Module):
    """
    BERTClassifier classification model
    """
    def __init__(self, hidden_size, num_classes, num_layers, bidirectional, dropout_prob=0.3):
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
#         print(input_ids)
        embedded = outputs[0].to(device)
#         print(embedded.shape)
        lstm, _ = self.lstm(embedded)
        lstm = lstm.to(device)
#         print(lstm.shape)
        num_embedded = (input_ids==0).float().sum(1)
#         print(num_embedded)
        pooled = self.pool(lstm).to(device)
#         print(pooled.shape)
#         print(num_embedded)
#         print(num_embedded)



        for i in range(num_embedded.shape[0]):
            if num_embedded[i] == 0:
                num_embedded[i] = 1
#         print(num_embedded)
        average_embedding = pooled.sum(1) / num_embedded.view(-1,1) 
        
#         embedded = self.dropout(embedded).to(device)

#         linear = self.linear_layer(embedded).to(device)
#         print(average_embedding)
        linear = self.clf(average_embedding).to(device)
        
        
#         print(linear)
        non_linear = self.non_linearity(linear).to(device)
    
        logits = F.softmax(non_linear, dim=1).to(device)
        
        return logits



def evaluate(model, dataloader, threshold, num_labels):
    f1_res = []
    acc_res = []
    model.eval()
    """
        4. TODO: Your code here
        Calculate the accuracy of the model on the data in dataloader
        You may refer to `run_inference` function from Lab2 
    """
    with torch.no_grad():
        all_preds = []
        labels = []
        for batch_text, batch_labels in dataloader:
            labels += batch_labels.detach().cpu().tolist()
            preds_batch = model(batch_text)
#             print("batch:", preds_batch)
            for preds in preds_batch:
                for i in range(len(preds)):
                    if preds[i] >= threshold:
                        preds[i] = 1
                    else:
                        preds[i]= 0
                all_preds.append(preds.detach().cpu().tolist())
                
        labels = np.vstack(labels)
        all_preds = np.vstack(all_preds)
#         print(labels)
#         print(all_preds)
        for i in range(num_labels):
#                 print(batch_labels)
            f1 = f1_score(labels[:,i], all_preds[:,i])
            acc = accuracy_score(labels[:,i], all_preds[:,i])
        
            f1_res.append(f1)
            acc_res.append(acc)

    return acc_res, f1_res




num_labels = 4
hidden_size = 64
num_layers = 1
bidirectional = True
model = LSTMClassifier(hidden_size, num_labels, num_layers, bidirectional)
model.to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.03)



train_loss_history = []
val_accuracy_history = []
val_f1_history = []
best_val_accuracy = 0
n_no_improve = 0
early_stop_patience=7
NUM_EPOCHS=8
curr_max = 0
threshold = 0.35

for epoch in tqdm.tqdm(range(NUM_EPOCHS)):
    if n_no_improve == early_stop_patience:
        break
    else:
        model.train() # this enables regularization, which we don't currently have
        for i, (data_batch, batch_labels) in enumerate(train_loader):
            """
               Code for training lstm
               Keep track of training of for each batch using train_loss_history
            """
#             print(data_batch)
#             print(batch_labels)
            # data_batch = data_batch.to(device)
            # batch_labels = batch_labels.type(torch.FloatTensor)
            # batch_labels = batch_labels.to(device)
            preds = model(data_batch)
            
            loss = criterion(preds, batch_labels)
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

        accuracy, f1 = evaluate(model, val_loader, threshold, num_labels)
        
#         print("current accuracy: ", accuracy)
#         print("current f1: ", f1)
        
        val_accuracy_history.append(accuracy)
        val_f1_history.append(f1)
        
        if len(val_accuracy_history) == 1:
            curr_max = val_accuracy_history[0]
            torch.save(model, 'best_model.pt')
        elif len(val_accuracy_history) > 1:
            if accuracy > curr_max:
                curr_max = accuracy
                torch.save(model, 'best_model.pt')
                n_no_improve = 0
            else:
                n_no_improve += 1
        best_val_accuracy = curr_max

print("acc:", val_accuracy_history)
print("f1:", val_f1_history)
print("loss", train_loss_history)
# print("Best validation accuracy is: ", best_val_accuracy)

