import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import pickle
from nltk.tokenize import WordPunctTokenizer

# BERT
from transformers import BertTokenizer

# CRF
from er_pipeline.crf.crf_utils import prepare_prompt

def parse_tag(tag):
    """
    Parses a tag into a tuple (tag, IOB).
    """
    return (None, tag) if tag in {'PAD', 'O'} else tuple(tag.split('-'))

def clean_prediction_sequence(predictions):
    """
    Enforces BIO rules. 
    """
    if not predictions:
        return

    # if the first tag is I, change it to B
    try:
        curr_prefix, curr_tag = parse_tag(predictions[0])
    except ValueError as e:
        raise ValueError(f"ValueError: {predictions[0]}") from e

    if curr_prefix == 'I':
        predictions[0] = 'B-' + curr_tag

    for prev_idx, tag in enumerate(predictions[1:]):

        if tag not in {'PAD'}:

            curr_prefix, curr_tag = parse_tag(tag)

            if predictions[prev_idx] in {'PAD'} and curr_prefix == 'I':
                predictions[prev_idx + 1] = 'B-' + curr_tag

            elif predictions[prev_idx] not in {'PAD'}:
                _, prev_tag = parse_tag(predictions[prev_idx])

                if curr_prefix == 'I' and prev_tag != curr_tag:
                    predictions[prev_idx + 1] = 'B-' + curr_tag

    return predictions

class SpanDetector():
    
    def __init__(self, model, config) -> None:
        
        self.model_name = model
        self.config = config
        
        if self.model_name == 'bert': 
            
            self.load_bert()
            
        elif self.model_name == 'crf':
            
            self.load_crf()
        
    def load_bert(self):
        
        # load model weights
        tag2idx = np.load(self.config['model']['bert']['tag2idx'], allow_pickle = 'TRUE').item()
        self.idx2tag = {v: k for k, v in tag2idx.items()}
        
        self.model = torch.load(self.config['model']['weights'], map_location = torch.device("cuda:0")).cuda()
        self.model.eval()
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case = False)

    def load_crf(self):
                
        # load model weights
        self.model = pickle.load(open(self.config['model']['crf']['saved_model'], 'rb'))
        
    def get_tokens(self, text):
        
        if self.model_name == 'bert':
        
            tokenized_text = self.tokenizer(text.lower(), return_tensors = 'pt')
            
            return self.tokenizer.convert_ids_to_tokens(tokenized_text['input_ids'][0], skip_special_tokens = True)

        elif self.model_name == 'crf':
            
            return WordPunctTokenizer().tokenize(text)

    def get_spans(self, text, predictions):
        # sourcery skip: my-custom-rule, raise-specific-error
        """
        Retrieves the span dict from tag prediction list.
        """
        spans = {key: [] for key in {'Skill', 'Experience', 'Occupation', 'Domain', 'Qualification'}}
        
        if not predictions:
            return spans
        
        clean_predictions = clean_prediction_sequence(predictions)
        tokens = self.get_tokens(text)

        rolling_span_capture = []
        current_tag = None

        for idx, tag in enumerate(clean_predictions):
            
            if tag not in {'PAD'}:

                prefix, label = parse_tag(tag)

                # print(f"prefix: {prefix}, label: {label}, token: {tokens[idx]}") # DEBUG

                # if it's I, continue listening
                if prefix == 'I':
                    if current_tag is None:
                        raise Exception(f"Something wrong here: {clean_predictions}")
                    rolling_span_capture.append(tokens[idx])

                elif rolling_span_capture:
                    spans[current_tag].append(self.tokenizer.convert_tokens_to_string(rolling_span_capture))
                    rolling_span_capture = []
                    current_tag = None

                # then, if it's B, listen to new
                if prefix == 'B':
                    rolling_span_capture.append(tokens[idx])
                    current_tag = label

        return spans
        
    def predict(self, text):
        
        if self.model_name == 'bert':
        
            tokenized_input = self.tokenizer(text.lower(), return_tensors = 'pt').to("cuda:0")
            labels = torch.tensor([1] * tokenized_input['input_ids'].size(1)).unsqueeze(0).to("cuda:0")

            output = self.model(**tokenized_input, labels = labels)
            logits = output.logits.detach().cpu().numpy()

            predictions_idx = [list(p) for p in np.argmax(logits, axis = 2)][0][1:-1]

            return [self.idx2tag[idx] for idx in predictions_idx]
        
        elif self.model_name == 'crf':
            
            X_test = prepare_prompt(text)
            return self.model.predict(X_test)[0]
    
    def predict_spans(self, text):
        
        predictions = self.predict(text)
        
        return self.get_spans(text, predictions)
    
def combine_span_dicts(span_dict_a, span_dict_b):
    """
    Combines two dicts of extracted spans. 
    """
    output_dict = {}
    for key in {'Skill', 'Experience', 'Occupation', 'Domain', 'Qualification'}:
        if key in span_dict_a and key not in span_dict_b:
            output_dict[key] = span_dict_a[key]
        elif key not in span_dict_a and key in span_dict_b:
            output_dict[key] = span_dict_b[key]
        else:
            output_dict[key] = list(set(span_dict_a[key]).union(set(span_dict_b[key])))
    return output_dict
    
    
def create_extracted_span_dict(df, id_field, span_detector):
    """
    Creates a dictionary of extracted spans from a dataframe.
    Assumes the dataframe has a 'combined' column. 
    """
    
    raw_to_extracted_span_dict = {}
    
    for row in tqdm(df.itertuples(), total = len(df), desc = 'Applying ER model'):
        
        if getattr(row, 'combined') not in raw_to_extracted_span_dict:
            
            cumulative_extracted_span_dict = {}
            
            for sentence in getattr(row, 'combined').split('.'):
                
                cumulative_extracted_span_dict = combine_span_dicts(cumulative_extracted_span_dict, span_detector.predict_spans(sentence))
            
            raw_to_extracted_span_dict[getattr(row, 'combined')] = cumulative_extracted_span_dict
            
    return raw_to_extracted_span_dict

def create_extracted_span_df(df, id_field, span_detector):
    """
    Creates a dataframe of extracted spans from a dataframe.
    Assumes the dataframe has a 'combined' column. 
    """
    
    raw_to_extracted_span_dict = create_extracted_span_dict(df, id_field, span_detector)
    
    extracted_span_df = df[[id_field, 'combined']].copy()
    extracted_span_df['extracted_spans'] = extracted_span_df['combined'].map(raw_to_extracted_span_dict)
    
    return extracted_span_df[[id_field, 'extracted_spans']]

def load_job_data(config):
        """
        Loads job data.
        Abstracted from TribePad-Matched-Data.utils.utils DataTable.       
        """

        line_count = sum(
            line.count(
                config['delimiters']['lineterminator']
            ) for line in open(config['input_data'], 'r')
        ) + 1 # account for last line with no terminator

        chunksize = round(line_count / 100)
        if chunksize < 10:
            return pd.read_csv(
                            config['input_data'], 
                            engine = 'c',
                            sep = config["delimiters"]['sep'],
                            lineterminator = config["delimiters"]['lineterminator'],
                            quoting=3, # to fix EOF error
                            # skiprows=0, # skip header
                            names=config['colnames'], 
                            error_bad_lines=False
                            )
        else:
            return pd.concat(
                list(
                    tqdm(
                        pd.read_csv(
                            config['input_data'], 
                            engine = 'c',
                            sep = config["delimiters"]['sep'],
                            lineterminator = config["delimiters"]['lineterminator'],
                            chunksize=chunksize,
                            quoting=3, # to fix EOF error
                            # skiprows=0, # skip header
                            names=config['colnames'], 
                            error_bad_lines=False
                            ), 
                        desc="Loading...", 
                        total=100
                        )
                    )
                )