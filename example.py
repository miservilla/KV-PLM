# Import the necessary components from your module
from smtokenization import SmilesTokenizer, get_default_tokenizer, convert_reaction_to_valid_features

# Instantiate the tokenizer
tokenizer = get_default_tokenizer()

# Load and process your SMILES data
with open("data/smiles.txt", "r") as file:
    for line in file:
        smiles = line.strip()
        tokenized_smiles = tokenizer.tokenize(smiles)

        # Print the tokenized SMILES string
        print("Original SMILES:", smiles)
        print("Tokenized SMILES:", tokenized_smiles)

        # Optional: Convert to valid features
        # features = convert_reaction_to_valid_features(smiles, tokenizer)

        # Here you would add your code to process these features,
        # such as feeding them into a machine learning model
