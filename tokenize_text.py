from tokenization import BertTokenizer  # Adjust this import based on the actual content of tokenization.py
from transformers import BertTokenizer, BertForPreTraining
from smtokenization import SmilesTokenizer
import torch
import numpy as np

# Path to your text file
text_file_path = 'data/test.txt'

# Initialize the tokenizer
# tokenizer = BertTokenizer.from_pretrained('scibert-large-uncased')  # Adjust the model as needed
tokenizer = SmilesTokenizer(vocab_file='vocab/vocab_all.txt')
# tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

# Read and tokenize the text file
tokenized_data = []
with open(text_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        tokenized_line = tokenizer.tokenize(line.strip())
        tokenized_data.append(tokenized_line)



# Now tokenized_data contains the tokenized lines of your text file

# Further processing...
# For example, you could write the tokenized data to another file
with open('data/test_output.txt', 'w', encoding='utf-8') as out_file:
    for tokenized_line in tokenized_data:
        out_file.write(' '.join(tokenized_line) + '\n')

with open('data/test_output.txt', 'r') as file:
    content = file.read()


SM = content

inp_SM = tokenizer.encode(SM)#[i+30700 for i in tokenizer.encode(SM)]
inp_SM = inp_SM[:min(128, len(inp_SM))]
inp_SM = torch.from_numpy(np.array(inp_SM)).long().unsqueeze(0)
att_SM = torch.ones(inp_SM.shape).long()
print(inp_SM)