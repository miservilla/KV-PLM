from tokenization import BertTokenizer  # Adjust this import based on the actual content of tokenization.py

# Path to your text file
text_file_path = 'data/test.txt'

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Adjust the model as needed

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
