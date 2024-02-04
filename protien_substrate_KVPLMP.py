from transformers import BertTokenizer, BertForPreTraining
import sys
import torch
import torch.nn as nn
import numpy as np

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
while True:
    SM = input("SMILES string: ")
    #txt = input("description: ")
    inp_SM = tokenizer.encode(SM)#[i+30700 for i in tokenizer.encode(SM)]
    inp_SM = inp_SM[:min(128, len(inp_SM))]
    inp_SM = torch.from_numpy(np.array(inp_SM)).long().unsqueeze(0)
    att_SM = torch.ones(inp_SM.shape).long()

    #inp_txt = tokenizer.encode(txt)
    #inp_txt = inp_txt[:min(128, len(inp_txt))]
    #inp_txt = torch.from_numpy(np.array(inp_txt)).long().unsqueeze(0)
    #att_txt = torch.ones(inp_txt.shape).long()

    if if_cuda:
        inp_SM = inp_SM.cuda()
        att_SM = att_SM.cuda()
        #inp_txt = inp_txt.cuda()
        #att_txt = att_txt.cuda()

    with torch.no_grad():
        #logits_des = model(inp_txt, att_txt, if_cuda)
        #embeddings= model.get_emb(inp_SM, att_SM, if_cuda)
        #score = torch.cosine_similarity(logits_des, logits_smi, dim=-1)
        #print('Matching score = ', score[0].item())
        #print('\n')
        embeddings = model.get_chosen_emb(inp_SM,att_SM)
        print(embeddings)
