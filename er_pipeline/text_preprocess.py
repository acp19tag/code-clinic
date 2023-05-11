import string
from bs4 import BeautifulSoup
import nlp
import unidecode
import contractions
import contextlib
from word2number import w2n
import spacy
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm import tqdm

def replace_bad_char(char):
    return char if char in string.ascii_letters + string.digits else ' '

def preprocess_text(text):
    """preprocesses text"""
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = ''.join(replace_bad_char(char) for char in text).replace('\n', '')  
    return ' '.join(text.split())

# Removes HTML tags from the text
def remove_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=' ')

# Removes extra whitespaces from the text
def remove_whitespace(text):
    text = text.strip()
    return " ".join(text.split())

# Converts the text to lowercase
def lowercase(text):
    return text.lower()

# Removes accented characters from the text
def remove_accented_char(text):
    return unidecode.unidecode(text)

# Expands the contractions within the text
def expand_contractions(text):
    return contractions.fix(text)


class TextPreprocessor():
    
    def __init__(self) -> None:
        
        self.nlp = spacy.load('en_core_web_sm')
        self.detokenizer = TreebankWordDetokenizer()
        
    # Processes and cleans the input text as specified by the arguments
    def preprocess(self, text, remove_htmltags=True, remove_extra_whitespace=True,
                remove_accent=True, remove_contractions=True,
                convert_lowercase=True, stop_words=False, punctuations=False,
                special_chars=True, remove_num=False, convert_num=False,
                lemmatization=False):
        # Call the necessary functions to perform cleaning
        if remove_htmltags:
            text = remove_html(text)

        if remove_extra_whitespace:
            text = remove_whitespace(text)

        if remove_accent:
            text = remove_accented_char(text)

        if remove_contractions:
            text = expand_contractions(text)

        if convert_lowercase:
            text = lowercase(text)

        # Use Spacy nlp() to tokenize the text
        doc = self.nlp(text)

        cleaned_text = []

        # CHeck whether each token belongs to any of the category to be removed,
        # which are specified by the function arguments
        for token in doc:
            flag = True
            token_text = token.text

            if stop_words and token.is_stop and token.pos_ != 'NUM':
                flag = False

            if punctuations and token.pos_ == 'PUNCT' and flag:
                flag = False

            if special_chars and token.pos_ == 'SYM' and flag:
                flag = False

            if remove_num and (token.pos_ == 'NUM' or token.text.isnumeric()) and flag:
                flag = False

            with contextlib.suppress(Exception):
                if convert_num and token.pos_ == 'NUM' and flag:
                    token_text = w2n.word_to_num(token.text)
            if lemmatization and token.lemma_ != "-PRON-" and flag:
                token_text = token.lemma_

            # If flag is True, which means that the token does not belong to any category
            # to be removed, we append it to the cleaned text list.
            if token_text != "" and flag:
                cleaned_text.append(token_text)

        return self.detokenizer.detokenize(cleaned_text)

    def preprocess_col(self, df_col):
        """preprocesses a column of a dataframe"""
        process_dict = {}
        
        for row in tqdm(df_col, desc='Preprocessing text'): 
            if row not in process_dict:
                process_dict[row] = self.preprocess(row)
                
        return df_col.map(process_dict)
        