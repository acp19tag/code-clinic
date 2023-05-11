"""
Decomposable Attention model for TribePad Status Prediction NLI Task.
"""



from dataloader import *
from model import *
import argparse
import importlib
# import wandb

import warnings
warnings.filterwarnings("ignore", category=UserWarning) # addresses UserWarning: The given NumPy array is not writable

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system') # addresses Too Many Files Open Runtime Error

with open('embedding_config.json', 'r') as infile:
    implemented_embed_methods = json.load(infile)

############################
# ARGUMENT PARSING
############################

parser = argparse.ArgumentParser(description='Decomposable Attention model for TribePad Status Prediction NLI Task.')

parser.add_argument('--embedding_method', type=str, default='SentenceBERT')

args, _ = parser.parse_known_args()

data_dir = 'data/apps/'
entities_dir = 'data/encoded_entities/'

if args.embedding_method not in implemented_embed_methods:
    raise ValueError(f"Embedding method {args.embedding_method} not supported. Supported methods are: {', '.join(implemented_embed_methods)}.")

############################
# EMBEDDING
############################

embed_size = implemented_embed_methods[args.embedding_method]['embed_size']

############################
# PARAMETERS
############################

batch_size = 256*4 # originally 256
num_hiddens = embed_size*2 # originally 200 
num_inputs_attend = embed_size # originally 100
num_inputs_compare = embed_size*2 # originally 200
num_inputs_agg = embed_size*4 # orignally 400
lr = 1e-4
num_epochs = 10 # prev 200
dropout = 0.3 # prev 0.2

early_stopping = True
patience = 20
devices = try_all_gpus()

############################
# WANDB INITIALISATION - not required on synthetic data
############################

# wandb.login()

# run = wandb.init(
#     project = 'job-matching-pipeline',
#     config = {
#         'batch_size': batch_size,
#         'learning_rate': lr,
#         'dropout': dropout,
#         'data': 'synthetic',
#         'embed_method': args.embedding_method,
#         'embed_size': embed_size,
#         'epochs': num_epochs,
#         'architecture': 'Decomposable Attention'
#     }
# )

############################
# LOAD DATA
############################

train_iter, dev_iter, test_iter = load_encoded_data_TribePad(
    data_dir = data_dir, 
    encoded_data_dir = entities_dir,
    data_type = None,
    batch_size = batch_size,
    # num_steps = num_steps
    )

############################
# INITIALISE MODEL
############################

net = DecomposableAttention(
    num_hiddens = num_hiddens, 
    num_inputs_attend = num_inputs_attend,
    num_inputs_compare = num_inputs_compare,
    num_inputs_agg = num_inputs_agg,
    dropout = dropout,
    root_data_dir = entities_dir,
    )

print(net)

# glove_embedding = TokenEmbedding(embed_model)
# embeds = InputEmbedding[vocab.idx_to_token]
# net.embedding.weight.data.copy_(embeds);

trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
report = train(
    net, 
    train_iter, 
    dev_iter, 
    test_iter, 
    loss, 
    trainer, 
    num_epochs, 
    devices, 
    wandb = None,
    early_stopping = early_stopping, 
    patience = patience
    )

# wandb.log(report)

print(report)