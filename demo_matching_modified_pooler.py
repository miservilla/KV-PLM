from transformers import BertTokenizer, BertForPreTraining
import sys
import torch
import torch.nn as nn
import numpy as np
from transformers import BertForSequenceClassification, BertConfig


tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

class BigModel(nn.Module):
    def __init__(self, main_model):
        super(BigModel, self).__init__()
        self.main_model = main_model
        self.dropout = nn.Dropout(0.1)

    def forward(self, tok, att, cud=True):
        typ = torch.zeros(tok.shape).long()
        if cud:
            typ = typ.cuda()
        pooled_output = self.main_model(tok, token_type_ids=typ, attention_mask=att)['pooler']
        logits = self.dropout(pooled_output)
        return logits
    
    def get_emb(self, tokens, token_type_ids, attention_mask):
        pooled_output = self.main_model(tokens, token_type_ids, attention_mask)['pooler']
        return pooled_output

#bert_model0 = BertForPreTraining.from_pretrained('allenai/scibert_scivocab_uncased')
bert_model = BertForPreTraining.from_pretrained('allenai/scibert_scivocab_uncased')
#bert_model.classifier = nn.Linear(768,13)

pt = torch.load('save_model/ckpt_KV.pt')
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

model = BigModel(bert_model)

if_cuda = False
if if_cuda:
    # model.load_state_dict(torch.load('save_model/ckpt_KV.pt'))
    # model = model.cuda()
    model = model.cuda()


print("Finished loading ckpt_kv")
model.eval()
#while True:
    # SM = input("SMILES string: ")
#    txt = input("description: ")
    # inp_SM = tokenizer.encode(SM)#[i+30700 for i in tokenizer.encode(SM)]
    # inp_SM = inp_SM[:min(128, len(inp_SM))]
    # inp_SM = torch.from_numpy(np.array(inp_SM)).long().unsqueeze(0)
    # att_SM = torch.ones(inp_SM.shape).long()

#    inp_txt = tokenizer.encode(txt)
#    inp_txt = inp_txt[:min(128, len(inp_txt))]
#    inp_txt = torch.from_numpy(np.array(inp_txt)).long().unsqueeze(0)
#    att_txt = torch.ones(inp_txt.shape).long()

#    if if_cuda:
        # inp_SM = inp_SM.cuda()
        # att_SM = att_SM.cuda()
#        inp_txt = inp_txt.cuda()
#        att_txt = att_txt.cuda()

    # with torch.no_grad():
    #     logits_des = model(inp_txt, att_txt, if_cuda)
    #     # logits_smi = model(inp_SM, att_SM, if_cuda)
    #     # score = torch.cosine_similarity(logits_des, logits_smi, dim=-1)
    #     # print('Matching score = ', score[0].item())
    #     print(logits_des)
    #     print('\n')

#    with torch.no_grad():
#        pooled_output = model.get_emb(inp_txt, torch.zeros(inp_txt.shape).long().cuda() if if_cuda else torch.zeros(inp_txt.shape).long(), att_txt)
#        print(pooled_output)
#        print('\n')
#        print(pooled_output.shape)
