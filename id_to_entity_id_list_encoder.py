"""

Script that:
    - generates a memmap for entity ids:
        e.g. input: user_id 00001:
            lookup user_id index in config
            return entity id list (e.g. [1, 2, 3, 4])
            each entity id can be looked up in separate config
            e.g. entity id 1: [0.0001, 0.0001, etc] <- encoded
        
Generates: 
    - config.json: a json file containing the configuration of the encoding
    - entity_id.memmap: a numpy memmap file containing the entity ids
    - entities.memmap: a numpy memmap file containing the encoded entities
    - id_to_entity_id_memmap_index.json: a json file containing the mapping from user/job ids to indices in the entities.memmap file

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
from sentence_transformers import SentenceTransformer

################
# PARAMETERS
################

root_app_data_dir = 'data/apps/'
user_entities_dir = 'data/user_extracted_spans.csv'
job_entities_dir = 'data/job_extracted_spans.csv'
embedding_method_config_dir = 'embedding_config.json'
excluded_entity_types = set()
root_output_dir = 'data/encoded_entities/'

entity_length_cap = 128 # max is minimum of this value and the max length in the training data

################
# EMBEDDING METHOD CONFIG
################

with open(embedding_method_config_dir, 'r') as infile:
    embedding_method_config = json.load(infile)

################
# ARGUMENT PARSING
################

parser = argparse.ArgumentParser(description='Reformat user data for ER pipeline tasks.')

parser.add_argument('--test', action='store_true', help='use dummy data')
parser.add_argument('--data', type = str, help='hired or interviewed')
parser.add_argument('--embedding_method', type=str, help='embedding method to use for encoding')

args, _ = parser.parse_known_args()

if args.embedding_method not in embedding_method_config:
    raise ValueError('embedding method not found in config. Please choose from: ' + ', '.join(embedding_method_config.keys()))

################
# START TIMER
################

timer = TimeManager()

################
# EMBEDDING PROTOCOL
################

print('Loading embedding model...')

model = SentenceTransformer(args.embedding_method)
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

# NEW FOR CODE CLINIC
def get_user_entity_list_from_index(index, user_df):
    if index in user_df.index:
        return ast.literal_eval(user_df.loc[index].entities)
    # print(f'Index {index} not found in user_df')
    return {entity_type: [] for entity_type in {'Skill', 'Experience', 'Qualification', 'Domain', 'Occupation'}}

def get_user_entity_span(user_id, user_df, excluded_entity_types = set()):
    
    extracted_spans = get_user_entity_list_from_index(user_id, user_df)
    
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
        # len(get_user_entity_list_from_index(index, user_df)) for index in user_df.index 
        len(get_user_entity_span(index, user_df)) for index in user_df.index # NEW FOR CODE CLINIC
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
        return {index: get_user_entity_span(index, user_df) for index in tqdm(list(user_ids)[:100], desc = 'Creating user id to entity dict: TEST MODE (100 users)')}
    return {index: get_user_entity_span(index, user_df) for index in tqdm(user_ids, desc = 'Creating user id to entity dict')} # NEW FOR CODE CLINIC
    
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
     
def entity_dict_to_entity_id_dict(
    entity_dict, 
    existing_entity_to_id_dict,
    max_entities,
    padding_index = 0,
    test = False
    ):
    """
    Takes an entity dict and optionally an existing entity to id dict, 
    and returns: 
    - an entity to entity id dict (should be saved in json format for later analysis)
    - an id to list of entity ids dict (for use in the model)
    """
    if existing_entity_to_id_dict is None:
        existing_entity_to_id_dict = {}
        current_id = 1
    else:
        current_id = max(existing_entity_to_id_dict.values()) + 1
        
    output_id_to_entity_id_dict = {}
        
    for item_index, entity_list in entity_dict.items():
        item_index = int(item_index) # it's int64 by default, which is not json serializable
        output_id_to_entity_id_dict[item_index] = []
        for entity_text in entity_list:
            if entity_text not in existing_entity_to_id_dict:
                existing_entity_to_id_dict[entity_text] = current_id
                current_id += 1
            output_id_to_entity_id_dict[item_index].append(existing_entity_to_id_dict[entity_text])
        # truncate if necessary
        if len(output_id_to_entity_id_dict[item_index]) > max_entities:
            output_id_to_entity_id_dict[item_index] = output_id_to_entity_id_dict[item_index][:max_entities]
        # pad if necessary
        elif len(output_id_to_entity_id_dict[item_index]) < max_entities:
            output_id_to_entity_id_dict[item_index] = output_id_to_entity_id_dict[item_index] + [padding_index] * (max_entities - len(output_id_to_entity_id_dict[item_index]))
                
        if test and current_id % 10 == 0:
            
            # print(f'existing_entity_to_id_dict: {existing_entity_to_id_dict}') # DEBUG
            # print(f'output_id_to_entity_id_dict: {output_id_to_entity_id_dict}') # DEBUG
            
            return existing_entity_to_id_dict, output_id_to_entity_id_dict
        
    # print(f'existing_entity_to_id_dict: {existing_entity_to_id_dict}') # DEBUG
    # print(f'output_id_to_entity_id_dict: {output_id_to_entity_id_dict}') # DEBUG
        
    return existing_entity_to_id_dict, output_id_to_entity_id_dict
     
def save_entity_id_dict_as_memmap(
    user_id_to_entity_id_dict: dict,
    job_id_to_entity_id_dict: dict,
    root_output_dir: str, 
    max_entities: int
):
    """
    Saves the entity id dict as a memmap file with index configuration. 
    
    inputs: 
        user_entity_id_dict {user_id: [entity_id, entity_id, ...], ...}
        job_entity_id_dict {job_id: [entity_id, entity_id, ...], ...}
    
    outputs:
        id_to_entity_id_index.json {user: {user_id: memmap_index, ...}, {job: {job_id: memmap_index, ...}}}
        entity_id.memmap
    """
    
    id_to_entity_id_index = {'user': {}, 'job': {}}
    entity_id_memmap = np.memmap(os.path.join(root_output_dir, 'entity_id.memmap'), dtype = 'int32', mode = 'w+', shape = (len(user_id_to_entity_id_dict) + len(job_id_to_entity_id_dict) + 1, max_entities))
    
    for index, (user_id, entity_id_list) in enumerate(tqdm(user_id_to_entity_id_dict.items(), desc = 'Saving user entity id dict as memmap')):
        id_to_entity_id_index['user'][user_id] = index + 1
        entity_id_memmap[index] = np.array(entity_id_list)
        
    for index, (job_id, entity_id_list) in enumerate(tqdm(job_id_to_entity_id_dict.items(), desc = 'Saving job entity id dict as memmap')):
        id_to_entity_id_index['job'][job_id] = len(user_id_to_entity_id_dict) + index + 1
        entity_id_memmap[len(user_id_to_entity_id_dict) + index] = np.array(entity_id_list)
        
    # default/unknown value at index 0
    id_to_entity_id_index['user'][0] = 0
    id_to_entity_id_index['job'][0] = 0
    entity_id_memmap[0] = np.zeros(max_entities)
        
    # write changes to disk
    entity_id_memmap.flush()
        
    with open(os.path.join(root_output_dir, 'id_to_entity_id_memmap_index.json'), 'w+') as f:
        json.dump(id_to_entity_id_index, f)
     

def convert_list_to_padded_array(input_list: list, max_entities: int, pad_value = 0):
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
):
    # memmap[index] = convert_list_to_padded_array(encoded_entity_list, max_entities, padding_value)
    memmap[index] = np.array(encoded_entity_list)
    memmap.flush()


def encode_entity_id_dict(
    entity_to_entity_id_dict,
    model
    ):
    """
    Encodes the entity(text) to entity(id) dict to give new dict: 
        - key: entity(id)
        - value: entity(encoded)
        
    Note that padding has already been done at this point (func entity_dict_to_entity_id_dict)
    """
    
    return dict(
        zip(
            list(entity_to_entity_id_dict.values()),
            embed(list(entity_to_entity_id_dict.keys()), model)
        )
    )

def create_memmap_for_encoded_entities(
    entity_to_entity_id_dict,
    model,
    root_output_dir,
    embedding_dim, 
    test = False
):
    """
    Saves a memmap for the encoded entities. 
    The memmap index corresponds to the entity id.
    """
    
    output_shape = (
        len(entity_to_entity_id_dict) + 1, 
        embedding_dim
        )
    
    # memmap
    encoded_entity_memmap = np.memmap(
        f'{root_output_dir}entities.memmap', 
        dtype='float32',
        mode = 'w+',
        shape = output_shape
    )
    
    encoded_entity_id_dict = encode_entity_id_dict(entity_to_entity_id_dict, model)
    
    for entity_id, encoded_entity_list in tqdm(encoded_entity_id_dict.items(), desc = 'Encoding entities'):
        encode_and_flush(
            memmap = encoded_entity_memmap,
            index = entity_id,
            encoded_entity_list = encoded_entity_list,
        )
        
    # create one blank one for index 0: used for testing
    encoded_entity_memmap[0] = np.zeros(embedding_dim)
    encoded_entity_memmap.flush()
    
################
# MAIN
################

app_data_dir = root_app_data_dir # NEW FOR CODE CLINIC

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



print('Getting user_id -> entity(text) data...')
entity_to_entity_id_dict, user_id_to_entity_id_dict = entity_dict_to_entity_id_dict(
    create_user_id_to_entity_dict(
        app_data_dir = app_data_dir,
        user_entities_dir = user_entities_dir,
        test = args.test
        ),
    existing_entity_to_id_dict = None,
    max_entities = max_entity_length
)

print('Getting job_id -> entity(text) data...')
entity_to_entity_id_dict, job_id_to_entity_id_dict = entity_dict_to_entity_id_dict(
    create_job_id_to_entity_dict(
        app_data_dir = app_data_dir,
        job_entities_dir = job_entities_dir,
        excluded_entity_types = excluded_entity_types,
        test = args.test
        ),
    existing_entity_to_id_dict = entity_to_entity_id_dict,
    max_entities = max_entity_length
)

print('Saving entity_to_entity_id_dict...')
save_entity_id_dict_as_memmap(
    user_id_to_entity_id_dict = user_id_to_entity_id_dict,
    job_id_to_entity_id_dict = job_id_to_entity_id_dict,
    root_output_dir = root_output_dir,
    max_entities = max_entity_length
)

print('Creating memmap for encoded entities...')
create_memmap_for_encoded_entities(
    entity_to_entity_id_dict = entity_to_entity_id_dict,
    model = model, 
    root_output_dir = root_output_dir,
    embedding_dim = embedding_method_config[args.embedding_method]['embed_size'],
    test = args.test
)

print('Creating output config...')

with open(f'{root_output_dir}config.json', 'w+') as outfile:
    json.dump(
        {
            'max_entity_length': max_entity_length,
            'n_entities': len(entity_to_entity_id_dict) + 1,
            'n_ids': len(user_id_to_entity_id_dict) + len(job_id_to_entity_id_dict) + 1, # +1 for default value
            'embedding_method': args.embedding_method,
            'embedding_dim': embedding_method_config[args.embedding_method]['embed_size'],
            'excluded_entity_types': list(excluded_entity_types),
        },
        outfile,
    )

timer.end()