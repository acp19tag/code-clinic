"""

Creates synthetic data for testing purposes.

Generated files: 

- data/apps/train.csv [130_022 rows x 3 columns]
- data/apps/dev.csv [43_494 rows x 3 columns]
- data/apps/test.csv [43_250 rows x 3 columns]
- data/job_extracted_spans.csv [200_000 rows x 2 columns]
- data/user_extracted_spans.csv [4_200_000 rows x 2 columns]

"""



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

##########################
# CREATE DIRECTORIES
##########################

if not os.path.exists('data/apps/'):
    os.mkdir('data/')
    os.mkdir('data/apps/')
    

##########################
# GENERATING APPS DATA
##########################

# first, generate the apps data
# randomly select 250_000 users and jobs with replacement
app_df = pd.DataFrame(
    {
        'user_id': np.random.choice(np.arange(1, 4_200_000), size=250_000, replace=True),
        'job_id': np.random.choice(np.arange(1, 200_000), size=250_000, replace=True),
    }
)

# drop duplicate rows
app_df = app_df.drop_duplicates()

# add status label; half 0, half 1
app_df['status'] = np.random.choice([0, 1], size=app_df.shape[0], replace=True)

# split into train, dev, test
train_df, test_df = train_test_split(app_df, test_size=0.2)
test_df, dev_df = train_test_split(test_df, test_size=0.5)

# save to csv
train_df.to_csv('data/apps/train.csv', index=False)
dev_df.to_csv('data/apps/dev.csv', index=False)
test_df.to_csv('data/apps/test.csv', index=False)

##########################
# GENERATING EXTRACTED SPANS DATA
##########################

# get sample skill list from file
with open('skill_lists/skills.txt', 'r') as f:
    skills = f.read().splitlines()

skill_dict = dict(enumerate(skills))
    
# def generate_extracted_skill_span(skills):
#     """
#     Generates a random skill span from the list of skills.
#     """
#     # number of skills should be between 1 and 5 (arbitrary - just has to be inconsistent)
#     num_skills = np.random.randint(1, 6)
    
#     # randomly select skills
#     skill_list = np.random.choice(skills, size=num_skills, replace=False)
    
#     return "{'Occupation': [], 'Qualification': [], 'Skill': " + str(list(skill_list)) + ", 'Experience': [], 'Domain': []}"

def generate_skill_index_array(n_elements_total, skill_list_length):
    
    # start by assuming each array has 3 skills
    
    long_index_list = np.random.choice(np.arange(0, skill_list_length), size=n_elements_total*3, replace=True)
    
    # reshape to 3 columns
    
    return long_index_list.reshape(-1, 3)
    
def lookup_array_to_skill_list(lookup_array, skill_dict):
    return [skill_dict[i] for i in lookup_array]

def convert_skill_list_to_output_format(skill_list):
    return "{'Occupation': [], 'Qualification': [], 'Skill': " + str(skill_list) + ", 'Experience': [], 'Domain': []}"    

job_skills_index_array = generate_skill_index_array(200_000, len(skills))
user_skills_index_array = generate_skill_index_array(4_200_000, len(skills))

# print(f"len np arange: {len(np.arange(1, 200_000))}") # DEBUG
# print(f"len converted skills: {len([convert_skill_list_to_output_format(lookup_array_to_skill_list(job_skills_index_array[i], skill_dict)) for i in range(200_000)])}") # DEBUG

job_spans_df = pd.DataFrame(
    {
        'job_id': np.arange(1, 200_000),
        'extracted_spans': [convert_skill_list_to_output_format(lookup_array_to_skill_list(job_skills_index_array[i], skill_dict)) for i in tqdm(range(200_000-1), desc = 'Creating job spans')]
    }
)

job_spans_df.to_csv('data/job_extracted_spans.csv', index=False)

user_spans_df = pd.DataFrame(
    {
        'user_id': np.arange(1, 4_200_000),
        'extracted_spans': [convert_skill_list_to_output_format(lookup_array_to_skill_list(user_skills_index_array[i], skill_dict)) for i in tqdm(range(4_200_000-1), desc = 'Creating user spans')]
    }
)

user_spans_df.to_csv('data/user_extracted_spans.csv', index=False)