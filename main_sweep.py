"""
Pipeline model with sweep for TribePad Status Prediction NLI Task.

Note: currently only configured for Decomposable Attention model.
"""


from dataloader import *
from model import *
import argparse
import importlib
import wandb
import json

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

parser.add_argument('--data', type = str, help='hired or interviewed')
parser.add_argument('--embedding_method', type=str, )

args, _ = parser.parse_known_args()

data_dir = '../data/matched/pipeline/apps/'
entities_dir = f'../data/matched/pipeline/encoded_entities/{args.data}/'

if args.embedding_method not in implemented_embed_methods:
    raise ValueError(f"Embedding method {args.embedding_method} not supported. Supported methods are: {', '.join(implemented_embed_methods.keys())}.")

############################
# EMBEDDING
############################

embed_size = implemented_embed_methods[args.embedding_method]['embed_size']

############################
# WANDB INITIALISATION
############################

sweep_configuration = {
    'method': 'grid',
    'name': 'pipeline-sweep',
    'metric': {'goal': 'maximize', 'name': 'dev_acc'},
    
    'parameters': {
        'batch_size': {
            # 'values': [256*2, 256*4]
            'value': 256*4
        },
        'learning_rate': {
            # 'values': [1e-4, 1e-5, 1e-6] # 1e-4 shown to be best in sweeps
            'value': 1e-4
        },
        'dropout': {
            'value': 0.3 # seems to be best in sweeps
        },
        'num_epochs': {'value': 200},
        'embedding_method': {'value': args.embedding_method},
        'embedding_size': {'value': embed_size},
        'architecture': {'value': 'Decomposable Attention'},
        'data': {'value': args.data},
    }
    
}

# static parameters
num_inputs_attend = embed_size # originally 100
num_hiddens = embed_size*2 # originally 200 
num_inputs_compare = embed_size*2 # originally 200
num_inputs_agg = embed_size*4 # orignally 400

early_stopping = True
patience = 20
devices = try_all_gpus()

sweep_id = wandb.sweep(
    sweep_configuration,
    project = 'job-matching-pipeline',
)

def main(config = None):
    
    with wandb.init(config = config):

        ############################
        # LOAD DATA
        ############################

        train_iter, dev_iter, test_iter = load_encoded_data_TribePad(
            data_dir = data_dir, 
            encoded_data_dir = entities_dir,
            data_type = args.data,
            batch_size = wandb.config.batch_size,
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
            dropout = wandb.config.dropout,
            root_data_dir = entities_dir,
            )

        trainer = torch.optim.Adam(net.parameters(), lr=wandb.config.learning_rate)
        loss = nn.CrossEntropyLoss(reduction="none")
        report = train(
            net, 
            train_iter, 
            dev_iter, 
            test_iter, 
            loss, 
            trainer, 
            wandb.config.num_epochs, 
            devices, 
            wandb = wandb,
            early_stopping = early_stopping, 
            patience = patience
            )

        wandb.log(report)
        
wandb.agent(sweep_id, function = main, count = 30)