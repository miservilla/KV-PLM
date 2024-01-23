from transformers import BertTokenizer, BertForPreTraining
import sys
import torch
import torch.nn as nn
import numpy as np
from transformers import BertForSequenceClassification, BertConfig
from smtokenization import SmilesTokenizer

tokenizer = SmilesTokenizer(vocab_file='vocab/vocab_all.txt')
# tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

class KvplmModel(nn.Module):
    def __init__(self, main_model,pool_type):
        super(KvplmModel, self).__init__()
        self.main_model = main_model
        self.pool_type = pool_type
    
    def forward(self,input_ids,attention_mask):
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

#bert_model0 = BertForPreTraining.from_pretrained('allenai/scibert_scivocab_uncased')
bert_model = BertForPreTraining.from_pretrained('allenai/scibert_scivocab_uncased')
#bert_model.classifier = nn.Linear(768,13)

pt = torch.load('save_model/ckpt_KV.pt',map_location=torch.device('cpu'))
if 'module.ptmodel.bert.embeddings.word_embeddings.weight' in pt:
    pretrained_dict = {k[20:]: v for k, v in pt.items()}
    #print('module')
    #print(pretrained_dict)
elif 'bert.embeddings.word_embeddings.weight' in pt:
    pretrained_dict = {k[5:]: v for k, v in pt.items()}
    #print('bert')
    #print(pretrained_dict)
else:
    pretrained_dict = {k[12:]: v for k, v in pt.items()}
    #print('none')
    #print(pretrained_dict)

bert_model.bert.load_state_dict(pretrained_dict, strict=False)

original_kvplm_bert = bert_model.bert

model = KvplmModel(original_kvplm_bert,'cls')
if_cuda = False
if if_cuda:
    # model.load_state_dict(torch.load('save_model/ckpt_KV.pt'))
    # model = model.cuda()
    model = model.cuda()


print("Finished loading ckpt_kv")
#model.eval()

while True:
    SM = input("SMILES string: ")
#    txt = input("description: ")
    inp_SM = tokenizer.encode(SM)#[i+30700 for i in tokenizer.encode(SM)]
    inp_SM = inp_SM[:min(128, len(inp_SM))]
    inp_SM = torch.from_numpy(np.array(inp_SM)).long().unsqueeze(0)
    att_SM = torch.ones(inp_SM.shape).long()
    print(inp_SM)

#    inp_txt = tokenizer.encode(txt)
#    inp_txt = inp_txt[:min(128, len(inp_txt))]
#    inp_txt = torch.from_numpy(np.array(inp_txt)).long().unsqueeze(0)
#    att_txt = torch.ones(inp_txt.shape).long()

#    if if_cuda:
        # inp_SM = inp_SM.cuda()
        # att_SM = att_SM.cuda()
#        inp_txt = inp_txt.cuda()
#        att_txt = att_txt.cuda()

    with torch.no_grad():
         embeddings = model(inp_SM,att_SM)
    #     # logits_smi = model(inp_SM, att_SM, if_cuda)
    #     # score = torch.cosine_similarity(logits_des, logits_smi, dim=-1)
    #     # print('Matching score = ', score[0].item())
         print(embeddings)
         print('\n')

#    with torch.no_grad():
#        pooled_output = model.get_emb(inp_txt, torch.zeros(inp_txt.shape).long().cuda() if if_cuda else torch.zeros(inp_txt.shape).long(), att_txt)
#        print(pooled_output)
#        print('\n')
#        print(pooled_output.shape)
