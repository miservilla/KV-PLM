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

# Load CSV file
df = pd.read_csv('ChEBI_id_name_SMILES_def_cleaned.csv')

# Initialize a list to hold embeddings
all_embeddings = []
tokens_file_path = 'def_generated_tokens.txt'

with open(tokens_file_path, 'w') as tokens_file:
    for index, row in df.iterrows():
        SM = row['Definition']  # Use the 'SMILES' column
        encoded_input = tokenizer(SM, add_special_tokens=True, return_tensors='pt')
        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']

        # Optionally, write the tokens to a file (token ids and tokens themselves)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        # Uncomment below to save as token.txt file
        # tokens_file.write(f"Row {index}: {' '.join(tokens)}\n")
        inp_SM = tokenizer.encode(SM, add_special_tokens=True)
        
        inp_SM = inp_SM[:min(128, len(inp_SM))]
        inp_SM = torch.tensor(inp_SM).unsqueeze(0)  # Updated for tensor creation
        att_SM = torch.ones(inp_SM.shape, dtype=torch.long)

        if if_cuda:
            inp_SM = inp_SM.cuda()
            att_SM = att_SM.cuda()

        with torch.no_grad():
            embeddings = model.get_chosen_emb(inp_SM, att_SM)
            # If using CUDA, ensure embeddings are on the GPU; otherwise, keep on CPU
            if if_cuda:
                embeddings = embeddings.cuda()
            all_embeddings.append(embeddings)  # Append the tensor directly without converting to numpy

# Convert list of embeddings to a tensor (optional, based on your needs)
embeddings_tensor = torch.stack(all_embeddings)

# Save embeddings
torch.save(embeddings_tensor, 'def_embeddings.txt')  # Saves the tensor to a file
