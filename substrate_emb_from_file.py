'''
I fixed the bugs and it is working .(This is only with retrieval checkpoint)
If you just want the embeddings without fine tuning then you can directly use 
get_chosen_emb function from BigModel class.

If you wanna fine tune you have to replace forward function with get_class_emb 
function (change name of get_class_emb to forward and delete the original 
forward.)

I dont think you need to use get_emb function. I am trying to follow what has 
been used for pooling in GeneLLM project
'''

from transformers import BertTokenizer, BertForPreTraining
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

if_cuda = False

tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

class BigModel(nn.Module):
    def __init__(self, main_model,pool_type):
        super(BigModel, self).__init__()
        self.main_model = main_model
        self.dropout = nn.Dropout(0.1)
        self.pool_type = pool_type

    def forward(self, tok, att, cud=True):
        typ = torch.zeros(tok.shape).long()
        if cud:
            typ = typ.cuda()
        pooled_output = self.main_model(tok, token_type_ids=typ, attention_mask=att)['pooler_output']
        logits = self.dropout(pooled_output)
        return logits

    def get_emb(self, tok, att, cud=True):
        typ = torch.zeros(tok.shape).long()
        if cud:
            typ = typ.cuda()
        pooled_output = self.main_model(tok, token_type_ids=typ, attention_mask=att)['pooler_output']
        return pooled_output
    
    def get_chosen_emb(self,input_ids,attention_mask):
        hiddenState, ClsPooled = self.main_model(input_ids = input_ids,attention_mask=attention_mask).values()
        if self.pool_type.lower() == "max":
            embeddings = self.max_pooling(hiddenState, attention_mask)
        elif self.pool_type.lower() == "cls":
            embeddings = ClsPooled
        elif self.pool_type.lower() == "mean":
            embeddings = self.mean_pooling(hiddenState, attention_mask)
        return embeddings
    
    def max_pooling(self, hidden_state, attention_mask):
        #CLS: First element of model_output contains all token embeddings
        token_embeddings = hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        pooled_embeddings = torch.max(token_embeddings, 1)[0]
        return pooled_embeddings
    
    def mean_pooling (self, hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        pooled_embeddings = torch.sum(hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9) 
        return pooled_embeddings



bert_model0 = BertForPreTraining.from_pretrained('allenai/scibert_scivocab_uncased')
model = BigModel(bert_model0.bert,"cls")
if if_cuda:
    model.load_state_dict(torch.load('save_model/ckpt_ret01.pt'))
    model = model.cuda()
else:
    model.load_state_dict(torch.load('save_model/ckpt_ret01.pt', map_location=torch.device('cpu') ))
model.eval()

df = pd.read_csv('/data_link/servilla/KV-PLM/compound_smiles_no_dups_Nan.csv')

# initial list to hold embeddings
all_embeddings = []
for cid in df['CID']:
    SM = df[df['CID'] == cid]['SMILES'].values[0]
    inp_SM = tokenizer.encode(SM)#[i+30700 for i in tokenizer.encode(SM)]
    inp_SM = inp_SM[:min(128, len(inp_SM))]
    inp_SM = torch.from_numpy(np.array(inp_SM)).long().unsqueeze(0)
    att_SM = torch.ones(inp_SM.shape).long()

    if if_cuda:
        inp_SM = inp_SM.cuda()
        att_SM = att_SM.cuda()

    with torch.no_grad():
        embeddings = model.get_chosen_emb(inp_SM, att_SM)
        all_embeddings.append(embeddings)

identifiers = df['CID'].values

# Convert all_embeddings to a list of numpy arrays (for human readability)
all_embeddings_np = [emb.cpu().numpy() for emb in all_embeddings]  # Make sure to move the tensor to CPU

# Flatten the embeddings if they are not already 1D
all_embeddings_flattened = [emb.flatten() for emb in all_embeddings_np]

# Create a DataFrame from the flattened embeddings
embeddings_df = pd.DataFrame(all_embeddings_flattened)

# Add the identifiers as the first column
embeddings_df.insert(0, 'Identifier', identifiers)

# Save the DataFrame to a CSV file
embeddings_df.to_csv('compound_smiles_emb_no_text.csv', index=False)

# while True:
#     SM = input("SMILES string: ")
#     #txt = input("description: ")
#     inp_SM = tokenizer.encode(SM)#[i+30700 for i in tokenizer.encode(SM)]
#     inp_SM = inp_SM[:min(128, len(inp_SM))]
#     inp_SM = torch.from_numpy(np.array(inp_SM)).long().unsqueeze(0)
#     att_SM = torch.ones(inp_SM.shape).long()

#     #inp_txt = tokenizer.encode(txt)
#     #inp_txt = inp_txt[:min(128, len(inp_txt))]
#     #inp_txt = torch.from_numpy(np.array(inp_txt)).long().unsqueeze(0)
#     #att_txt = torch.ones(inp_txt.shape).long()

#     if if_cuda:
#         inp_SM = inp_SM.cuda()
#         att_SM = att_SM.cuda()
#         #inp_txt = inp_txt.cuda()
#         #att_txt = att_txt.cuda()

#     with torch.no_grad():
#         #logits_des = model(inp_txt, att_txt, if_cuda)
#         #embeddings= model.get_emb(inp_SM, att_SM, if_cuda)
#         #score = torch.cosine_similarity(logits_des, logits_smi, dim=-1)
#         #print('Matching score = ', score[0].item())
#         #print('\n')
#         embeddings = model.get_chosen_emb(inp_SM,att_SM)
#         print(embeddings)
