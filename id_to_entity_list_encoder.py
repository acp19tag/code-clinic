"""

***OBSOLETE***

NOTE: This script is not compatible with DL2 architecture, since it does not include entity sequence IDs. 

Script that encodes the entities in the data set.

Generates: 
    - config.json: a json file containing the configuration of the encoding
    - job_entities.memmap: a numpy memmap file containing the encoded job entities
    - job_id_to_index.json: a json file containing the mapping from job ids to indices in the job_entities.memmap file
    - user_entities.memmap: a numpy memmap file containing the encoded user entities
    - user_id_to_index.json: a json file containing the mapping from user ids to indices in the user_entities.memmap file


usage: 

python -m entity_encoder [--test] [--interview] [--embedding_method EMBEDDING_METHOD]

"""

################
# IMPORTS
################

import os
import numpy as np
import argparse
import pandas as pd
import ast
import json
from utils.utils import TimeManager
from tqdm import tqdm
import random

################
# PARAMETERS
################

root_app_data_dir = '../data/matched/pipeline/apps/'
user_entities_dir = '../data/matched/pipeline/user_data.csv'
job_entities_dir = '../data/matched/pipeline/job_extracted_spans.csv'
excluded_entity_types = set()
root_output_dir = '../data/matched/pipeline/encoded_entities/'

entity_length_cap = 12 # max is minimum of this value and the max length in the training data

################
# EMBEDDING METHOD CONFIG
################

embedding_method_config = {
    'SentenceBERT': {
        'dim': 768,
    }
}

################
# ARGUMENT PARSING
################

parser = argparse.ArgumentParser(description='Reformat user data for ER pipeline tasks.')

parser.add_argument('--test', action='store_true', help='use dummy data')
parser.add_argument('--interview', action='store_true', help='use interview data')
parser.add_argument('--embedding_method', type=str, default='SentenceBERT', help='embedding method to use (only SentenceBERT implemented)')

args, _ = parser.parse_known_args()

################
# START TIMER
################

timer = TimeManager()

################
# EMBEDDING PROTOCOL
################

print('Loading embedding model...')
if args.embedding_method == 'SentenceBERT':
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    def embed(text_list, model = model):
        return model.encode(text_list)

################
# FUNCTIONS
################

def get_ids_from_train(train_data_dir):
    """Get the ids of the users and jobs in the training data."""
    train_df = pd.read_csv(train_data_dir)
    return {'user_ids': train_df.user_id.unique(), 'job_ids': train_df.job_id.unique()}

def get_ids_from_all(app_data_dir):
    """Get the ids of the users and jobs in all app data."""
    id_dict = {'user_ids': set(), 'job_ids': set()}
    for split in ['train', 'test', 'dev']:
        df = pd.read_csv(f'{app_data_dir}{split}.csv')
        id_dict['user_ids'].update(df.user_id.unique())
        id_dict['job_ids'].update(df.job_id.unique())
    return id_dict

def get_user_entity_list_from_index(index, user_df):
    if index in user_df.index:
        return ast.literal_eval(user_df.loc[index].entities)
    # print(f'Index {index} not found in user_df')
    return []

def convert_to_int(value):
    try:
        return int(float(value))
    except Exception:
        return 0
    
def get_job_entity_list_from_index(index, job_df):
    if index in job_df.index:
        return ast.literal_eval(job_df.loc[index].entities)
    # print(f'Index {index} not found in job_df')
    return {entity_type: [] for entity_type in {'Skill', 'Experience', 'Qualification', 'Domain', 'Occupation'}}

def get_job_entity_span(job_id, job_df, excluded_entity_types = set()):
    
    extracted_spans = get_job_entity_list_from_index(job_id, job_df)
    
    output_spans = []
    for entity_type in extracted_spans:
        if entity_type not in excluded_entity_types: 
            output_spans.extend(extracted_spans[entity_type])
            
    return output_spans

def find_max_entity_span_length_jobid(job_df):
    
    return max(
        len(get_job_entity_span(index, job_df)) for index in job_df.index
    )
    
def find_max_entity_span_length_userid(user_df):
    
    return max(
        len(get_user_entity_list_from_index(index, user_df)) for index in user_df.index
    )
    
def load_user_df(user_entities_dir):
    """Load the user data."""
    user_df = pd.read_csv(user_entities_dir, names = ['user_id', 'entities'])
    user_df['user_id'] = user_df['user_id'].apply(convert_to_int)
    user_df.set_index('user_id', inplace=True)
    if 0 in user_df.index:
        user_df = user_df.drop(0)
    return user_df

def load_job_df(job_entities_dir):
    """Load the job data."""
    job_df = pd.read_csv(job_entities_dir, skiprows = 2, names = ['job_id', 'entities'])
    job_df['job_id'] = job_df['job_id'].apply(convert_to_int)
    job_df.set_index('job_id', inplace=True)
    if 0 in job_df.index:
        job_df = job_df.drop(0)
    return job_df

def subset_df_by_id_set(df, id_set):
    """Subset a dataframe by a set of ids."""
    return df.loc[df.index.intersection(id_set)]

def find_max_length(train_data_dir, user_entities_dir, job_entities_dir):
    """Find maximum entity length in the training data."""
    
    train_id_dict = get_ids_from_train(train_data_dir)
    
    user_df = subset_df_by_id_set(
        load_user_df(user_entities_dir), 
        train_id_dict['user_ids']
    )
    
    job_df = subset_df_by_id_set(
        load_user_df(job_entities_dir),
        train_id_dict['job_ids']
    )
    
    return max(
        find_max_entity_span_length_userid(user_df),
        find_max_entity_span_length_jobid(job_df)
    )

def create_user_id_to_entity_dict(
    app_data_dir: str,
    user_entities_dir: str,
    test = False
):
    
    user_ids = get_ids_from_all(app_data_dir)['user_ids']
    user_df = load_user_df(user_entities_dir)
    if test:
        return {index: get_user_entity_list_from_index(index, user_df) for index in tqdm(list(user_ids)[:100], desc = 'Creating user id to entity dict: TEST MODE (100 users)')}    
    return {index: get_user_entity_list_from_index(index, user_df) for index in tqdm(user_ids, desc = 'Creating user id to entity dict')}
    
def create_job_id_to_entity_dict(
    app_data_dir, 
    job_entities_dir,
    excluded_entity_types = set(),
    test = False 
):
    job_ids = get_ids_from_all(app_data_dir)['job_ids']
    job_df = load_job_df(job_entities_dir)
    if test:
        return {index: get_job_entity_span(index, job_df, excluded_entity_types) for index in tqdm(list(job_ids)[:100], desc = 'Creating job id to entity dict: TEST MODE (100 jobs)')}
    return {index: get_job_entity_span(index, job_df, excluded_entity_types) for index in tqdm(job_ids, desc = 'Creating job id to entity dict')}
     
def encode_entity_dict(entity_dict, model):

    return {index: embed(entity_dict[index], model) for index in tqdm(entity_dict, desc = 'Encoding entity dict')}

def convert_list_to_padded_array(input_list: list, max_entities: int, pad_value = np.nan):
    """e.g. [[1,2], [4,5,6,7]] -> [[1,2,np.nan,np.nan], [4,5,6,7]]"""
    
    if not len(input_list):
        return np.full((1, max_entities), pad_value)
    elif len(input_list) > max_entities:
        return np.array(input_list[:max_entities])
    lens = np.array([len(item) for item in input_list])
    mask = lens[:,None] > np.arange(max_entities)
    out = np.full(mask.shape, pad_value)
    
    out[mask] = np.concatenate(input_list)
    return out

def pad_array_of_arrays(input_array_of_arrays, max_arrays: int, embed_dim: int, pad_value = np.nan):
    """e.g. [[[1,2], [4,5]], [[1,2], [4,5]]] -> [[[1,2], [4,5]], [[1,2], [4,5]], [[np.nan, np.nan], [np.nan, np.nan]]]"""
    # print(f'input_array_of_arrays.shape: {input_array_of_arrays.shape}') # DEBUG
    if len(input_array_of_arrays.shape) == 1:
        return np.full((max_arrays, embed_dim), pad_value)
    if input_array_of_arrays.shape[0] >= max_arrays:
        return input_array_of_arrays[:max_arrays]
    return np.append(input_array_of_arrays, np.full((max_arrays - input_array_of_arrays.shape[0], input_array_of_arrays.shape[1]), pad_value), axis = 0)

def convert_list_of_lists_to_padded_array(input_list_of_lists, max_top_list_len: int, max_sub_list_len: int, pad_value = np.nan):
    return np.array([
        pad_array_of_arrays(
            convert_list_to_padded_array(
                list_of_lists, max_sub_list_len, pad_value
            ),
            max_top_list_len,
            pad_value,
        )
        for list_of_lists in input_list_of_lists
    ])
    
def convert_default_memmap_value(memmap, shape, value):
    memmap[:] = value
    memmap.resize(shape)
    memmap.flush()
    
def encode_and_flush(
    memmap: np.memmap, 
    index: int, 
    encoded_entity_list: list, 
    max_entities: int, 
    embedding_dim: int,
    padding_value: int
):
    # memmap[index] = convert_list_to_padded_array(encoded_entity_list, max_entities, padding_value)
    memmap[index] = pad_array_of_arrays(encoded_entity_list, max_entities, embedding_dim, padding_value)
    memmap.flush()

def create_memmap_for_encoded_user_entities(
    app_data_dir: str, 
    user_entities_dir: str,
    root_output_dir: str,
    model,
    max_entities = 512,
    embedding_dim = 768,
    padding_value = 0,
    test = False    
):
    """Saves a memmap of the encoded user entities and an id to index mapping."""
    
    user_id_to_entity_dict = create_user_id_to_entity_dict(app_data_dir, user_entities_dir, test)
    user_id_to_entity_dict = encode_entity_dict(user_id_to_entity_dict, model)

    output_shape = (len(user_id_to_entity_dict), max_entities, embedding_dim)

    # memmap    
    encoded_entity_memmap = np.memmap(f'{root_output_dir}user_entities.memmap', dtype='float32', mode='w+', shape = output_shape)
    
    if padding_value != 0:
        convert_default_memmap_value(encoded_entity_memmap, output_shape, padding_value)

    for index, id in tqdm(enumerate(user_id_to_entity_dict), desc = 'Encoding user entities'):
        encode_and_flush(
            memmap = encoded_entity_memmap, 
            index = index, 
            encoded_entity_list = user_id_to_entity_dict[id], 
            max_entities = max_entities, 
            embedding_dim = embedding_dim, 
            padding_value = padding_value
            )

    # id to index mapping
    with open(f'{root_output_dir}user_id_to_index.json', 'w+') as outfile:
        
        # print(f'Ids: {list(user_id_to_entity_dict)}') # DEBUG

        json.dump(
            {int(id): index for index, id in enumerate(user_id_to_entity_dict)},
            outfile
        )
    
def create_memmap_for_encoded_job_entities(
    app_data_dir: str,
    job_entities_dir: str, 
    root_output_dir: str,
    model,
    max_entities = 512,
    embedding_dim = 768,
    padding_value = 0,
    excluded_entities = set(),
    test = False
): 
    """Save a memmap of the encoded job entities and an id to index mapping."""
    
    job_id_to_entity_dict = create_job_id_to_entity_dict(app_data_dir, job_entities_dir, excluded_entities, test)
    job_id_to_entity_dict = encode_entity_dict(job_id_to_entity_dict, model)
    
    output_shape = (len(job_id_to_entity_dict), max_entities, embedding_dim)
    
    # memmap    
    encoded_entity_memmap = np.memmap(f'{root_output_dir}job_entities.memmap', dtype='float32', mode='w+', shape = output_shape)

    if padding_value != 0:
        convert_default_memmap_value(encoded_entity_memmap, output_shape, padding_value)
        
    for index, id in tqdm(enumerate(job_id_to_entity_dict), desc = 'Encoding job entities'):
        encode_and_flush(
            memmap = encoded_entity_memmap, 
            index = index, 
            encoded_entity_list = job_id_to_entity_dict[id], 
            max_entities = max_entities, 
            embedding_dim = embedding_dim, 
            padding_value = padding_value
            )
    
    # id to index mapping
    with open(f'{root_output_dir}job_id_to_index.json', 'w+') as outfile:
        
        json.dump(
            {int(id): index for index, id in enumerate(job_id_to_entity_dict)},
            outfile
        )

################
# MAIN
################

data_type = 'interviewed' if args.interview else 'hired'

app_data_dir = f'{root_app_data_dir}{data_type}/'
root_output_dir = f'{root_output_dir}{data_type}/'

if not os.path.exists(root_output_dir):
    os.makedirs(root_output_dir)

if args.test:
    max_entity_length = 12
    print(f'Test mode: setting max entity length to {max_entity_length}')
else:
    print('Finding max entity length...')
    max_entity_length = find_max_length(f'{app_data_dir}train.csv', user_entities_dir, job_entities_dir)
if max_entity_length > entity_length_cap:
    print(f'Max entity length ({max_entity_length}) exceeds cap ({entity_length_cap}). Setting max entity length to {entity_length_cap}')
    max_entity_length = entity_length_cap

print('Creating output config...')
with open(f'{root_output_dir}config.json', 'w+') as outfile:
    json.dump(
        {
            'max_entity_length': max_entity_length,
            'embedding_method': args.embedding_method,
            'embedding_dim': embedding_method_config[args.embedding_method]['dim'],
            'excluded_entity_types': list(excluded_entity_types),
        },
        outfile,
    )

print('Creating memmap for user entities...')
create_memmap_for_encoded_user_entities(
    app_data_dir, 
    user_entities_dir,
    root_output_dir,
    model,
    max_entities = max_entity_length,
    embedding_dim = embedding_method_config[args.embedding_method]['dim'],
    test = args.test
)
print('Creating memmap for job entities...')
create_memmap_for_encoded_job_entities(
    app_data_dir, 
    job_entities_dir,
    root_output_dir,
    model,
    max_entities = max_entity_length,
    embedding_dim = embedding_method_config[args.embedding_method]['dim'],
    excluded_entities = excluded_entity_types,
    test = args.test
)

timer.end()