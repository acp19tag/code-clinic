import torch
from torch import nn
import pandas as pd
import re
import os
from collections import Counter
import hashlib
import requests
import zipfile
import tarfile
import time
import numpy as np
from IPython import display
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
from sklearn.metrics import classification_report
from tqdm import tqdm
import json

DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

DATA_HUB = {
    'glove.6b.50d': (
        f'{DATA_URL}glove.6B.50d.zip',
        '0b8703943ccdb6eb788e6f091b8946e82231bc4d',
    ),
    'glove.6b.100d': (
        f'{DATA_URL}glove.6B.100d.zip',
        'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a',
    ),
    'glove.42b.300d': (
        f'{DATA_URL}glove.42B.300d.zip',
        'b5116e234e9eb9076672cfeabf5469f3eec904fa',
    ),
    'wiki.en': (
        f'{DATA_URL}wiki.en.zip',
        'c1816da3821ae9f43899be655002f6c723e91b88',
    ),
}

#################################
# FUNCTIONS AND CLASSES
#################################

size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
to = lambda x, *args, **kwargs: x.to(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
transpose = lambda x, *args, **kwargs: x.t(*args, **kwargs)
reduce_mean = lambda x, *args, **kwargs: x.mean(*args, **kwargs)
expand_dims = lambda x, *args, **kwargs: x.unsqueeze(*args, **kwargs)
swapaxes = lambda x, *args, **kwargs: x.swapaxes(*args, **kwargs)
repeat = lambda x, *args, **kwargs: x.repeat(*args, **kwargs)


def tokenize(lines, token='word'):
    """Split text lines into word or character tokens."""
    assert token in ('word', 'char'), f'Unknown token type: {token}'
    return [line.split() if token == 'word' else list(line) for line in lines]

def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences."""
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad

def read_TribePad(data_dir, is_train):
    """Read the TribePad dataset."""
    
    def extract_text(s):
        # Remove information that will not be used by us
        s = re.sub('\\(', '', s)
        s = re.sub('\\)', '', s)
        # Substitute two or more consecutive whitespace with space
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()
    
    file_name = os.path.join(data_dir, 'train.txt'
                             if is_train else 'test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[0]) for row in rows]
    hypotheses = [extract_text(row[1]) for row in rows]
    labels = [int(row[2]) for row in rows]
    return premises, hypotheses, labels

def split_TribePad(premises, hypotheses, labels, dev_prop = 0.2):
    """Shuffles, then splits the TribePad dataset into train and dev sets."""
    num_examples = len(premises)
    indices = list(range(num_examples))
    np.random.shuffle(indices)
    dev_size = int(num_examples * dev_prop)
    train_indices = indices[dev_size:]
    dev_indices = indices[:dev_size]
    return ([premises[i] for i in train_indices], [hypotheses[i] for i in train_indices], [labels[i] for i in train_indices],
            [premises[i] for i in dev_indices], [hypotheses[i] for i in dev_indices], [labels[i] for i in dev_indices])

class Vocab:
    """Vocabulary for text."""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The list of unique tokens
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        return (
            [self.__getitem__(token) for token in tokens]
            if isinstance(tokens, (list, tuple))
            else self.token_to_idx.get(tokens, self.unk)
        )

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']

class TribePadDataset(torch.utils.data.Dataset):
    """A customised dataset to load the TribePad dataset."""
    
    def __init__(self, dataset, num_steps, vocab = None) -> None:
        self.num_steps = num_steps
        all_premise_tokens = tokenize(dataset[0])
        all_hypothesis_tokens = tokenize(dataset[1])
        if vocab is None:
            self.vocab = Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])
        print(f'read {len(self.premises)} examples')
        
    def _pad(self, lines):
        return torch.tensor([truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
    
def get_dataloader_workers():
    """Returns an appropriate number of workers for dataloader.
    WIP - currently returns 2"""
    # todo: scale with available resources

    return 2
    
def load_data_TribePad(data_dir, batch_size, num_steps = 50):
    """Return data iterators and vocabulary of the TribePad dataset."""
    
    num_workers = get_dataloader_workers()
    train_data = read_TribePad(data_dir, True)
    test_data = read_TribePad(data_dir, False)
    train_prem, train_hyp, train_labels, dev_prem, dev_hyp, dev_labels = split_TribePad(*train_data, dev_prop=0.2)
    train_set = TribePadDataset((train_prem, train_hyp, train_labels), num_steps)
    dev_set = TribePadDataset((dev_prem, dev_hyp, dev_labels), num_steps, train_set.vocab)
    test_set = TribePadDataset(test_data, num_steps, train_set.vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    dev_iter = torch.utils.data.DataLoader(dev_set, batch_size,
                                           shuffle = False, 
                                           num_workers = num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
    return train_iter, dev_iter, test_iter, train_set.vocab

def predict_TribePad(net, vocab, premise, hypothesis):
    """Predict whether the premise entails, contradicts, or is neutral to the
    hypothesis."""
    premise = torch.tensor(vocab[premise.split()], device=net.device)
    hypothesis = torch.tensor(vocab[hypothesis.split()], device=net.device)
    return torch.argmax(net(premise.unsqueeze(0), hypothesis.unsqueeze(0)),
                         dim=1)
    
def cpu():
    """Get the CPU device."""
    return torch.device('cpu')

def gpu(i=0):
    """Get a GPU device."""
    return torch.device(f'cuda:{i}')

def num_gpus():
    """Get the number of available GPUs."""
    return torch.cuda.device_count()

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    return gpu(i) if num_gpus() >= i + 1 else cpu()
    
def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    return [gpu(i) for i in range(num_gpus())]

def download(url, folder='../data', sha1_hash=None):
    """Download a file to folder and return the local filepath."""
    if not url.startswith('http'):
        # For back compatability
        url, sha1_hash = DATA_HUB[url]
    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, url.split('/')[-1])
    # Check if hit cache
    if os.path.exists(fname) and sha1_hash:
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                if data := f.read(1048576):
                    sha1.update(data)
                else:
                    break
        if sha1.hexdigest() == sha1_hash:
            return fname
    # Download
    if not os.path.exists(fname):    
        print(f'Downloading {fname} from {url}...')
        r = requests.get(url, stream=True, verify=True)
        with open(fname, 'wb') as f:
            f.write(r.content)
    return fname

def download_extract(name, folder=None):
    """Download and extract a zip/tar file."""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted.'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))

class Timer:
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

def use_svg_display():
    """Use the svg format to display a plot in Jupyter."""
    backend_inline.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib.
    """
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.
    """
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

class Animator:
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y, live = False):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        if live:
            display.display(self.fig)
            display.clear_output(wait=True)
            
    def save(self, fname):
        self.fig.savefig(fname)

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset using a GPU."""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            X = [x.to(device) for x in X] if isinstance(X, list) else X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), size(y))
    return metric[0] / metric[1]

def TribePad_classification_report(net, data_iter, device = 'cpu'):
    
    net.eval()
    if not device:
        device = next(iter(net.parameters())).device
            
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X, y in tqdm(data_iter, total = len(data_iter), desc = 'Evaluating'):
            X = [x.to(device) for x in X] if isinstance(X, list) else X.to(device)
            y = y.to(device)
            y_hat = net(X)
            y_hat = y_hat.argmax(axis=1)
            
            y_true.extend(y)
            y_pred.extend(y_hat)
            
    y_true = np.array([x.cpu() for x in y_true])
    y_pred = np.array([x.cpu() for x in y_pred])
            
    return classification_report(y_true, y_pred, output_dict = True)

def train_batch(net, X, y, loss, trainer, devices):
    """Train for a minibatch with multiple GPUs."""
    X = [x.to(devices[0]) for x in X] if isinstance(X, list) else X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def train(net, train_iter, dev_iter, test_iter, loss, trainer, num_epochs,
               devices=try_all_gpus(), wandb = None, early_stopping = False, patience = 5):
    """Train a model with multiple GPUs."""
    # timer, num_batches = Timer(), len(train_iter)
    # animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
    #                         legend=['train loss', 'train acc', 'dev acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    best_eval = 0
    plateau_count = 0
    for _ in tqdm(range(num_epochs), desc = 'Training'):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = Accumulator(4)
        for features, labels in train_iter:
            # timer.start()
            l, acc = train_batch(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            # timer.stop()
        dev_acc = evaluate_accuracy_gpu(net, dev_iter)
        # animator.add(epoch + 1, (None, None, dev_acc))
        if wandb is not None:
            wandb.log({'train_loss': metric[0] / metric[2], 'train_acc': metric[1] / metric[3], 'dev_acc': dev_acc}) # log metrics to wandb
        # print(f'epoch {epoch + 1:>3}, dev accuracy {dev_acc:.3f}')
        if dev_acc > best_eval:
            best_eval = dev_acc
            # print(f'New best model! Saving model at epoch {epoch + 1}.')
            torch.save(net.state_dict(), 'best_model.pt')
            plateau_count = 0
        elif early_stopping and plateau_count >= patience:
            net.load_state_dict(torch.load('best_model.pt'))
            return TribePad_classification_report(net, test_iter)
        else:
            plateau_count += 1

    net.load_state_dict(torch.load('best_model.pt'))
    return TribePad_classification_report(net, test_iter)

def evaluate(net, test_iter, devices = try_all_gpus()):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    print('Loading best model...')
    net.load_state_dict(torch.load('best_model.pt'))
    print('Evaluating best model...')
    return TribePad_classification_report(net, test_iter)    

class EntityLoader:
    """
    Revised entity loader that looks up the encoded version of the entity given its int id
    """
    
    def __init__(self, root_data_dir) -> None:
        
        # set config
        with open(f'{root_data_dir}config.json') as f:
            self.config = json.load(f)
            
        self.memmap = np.memmap(
            f'{root_data_dir}entities.memmap',
            dtype = 'float32',
            mode = 'r',
            shape = (self.config['n_entities'], self.config['embedding_dim'])
        )
        
    # def __getitem__(self, data_type, entity_id):
    #     return self.memmap[self.id_to_index[data_type][entity_id]]
        
    def get_entity(self, entity_id):
        try:
            return torch.tensor(self.memmap[entity_id].copy(), device = 'cuda') # TODO: don't hardcode device
        except KeyError:
            return torch.zeros((self.config['embedding_dim']), device = 'cuda') # TODO: don't hardcode device
    
    def get_entities(self, entity_list):
        """
        Note: entity_list is a torch tensor of entity ids.
        It is faster to do a batched lookup than to do a for loop (completes in ~5% of the time).
        """
        # FOR LOOP
        # return torch.stack([self.get_entity(entity_id.item()) for entity_id in entity_list.flatten()]).view(entity_list.shape + (self.config['embedding_dim'],))
    
        # BATCHED LOOKUP
        return torch.tensor(self.memmap[entity_list.cpu().flatten()].copy(), device = 'cuda').view(entity_list.shape + (self.config['embedding_dim'],))
    
class EncodedDataset(torch.utils.data.Dataset):
    """
    A customised dataset to load encoded entities from TribePad data.
    NOW modified to return entity ids instead of encoded entities. 
    
    """
    
    def __init__(
        self, 
        root_data_dir: str, 
        encoded_data_dir: str,
        data_type: str, 
        data_split: str, 
        n_ids: int,
        max_entity_length: int,
        ) -> None:
        """

        Args:
            root_data_dir (str): the root directory of the data files
            data_type (str): hired/interviewed
            data_split (str): train/test/dev
        """

        
        # load_data
        if data_type is not None:
            data_df = pd.read_csv(f'{root_data_dir}{data_type}/{data_split}.csv')
        else:
            data_df = pd.read_csv(f'{root_data_dir}{data_split}.csv')
        
        # load id to entity id memmap index dict 
        with open(f'{encoded_data_dir}/id_to_entity_id_memmap_index.json', 'r') as infile:
            self.id_to_entity_id_memmap_index = json.load(infile)
            
        # load entity_id memmap
        self.memmap = np.memmap(
            f'{encoded_data_dir}/entity_id.memmap',
            dtype = 'int32',
            mode = 'r',
            shape = (n_ids, max_entity_length)
        )

        self.users = data_df['user_id'].values
        self.jobs = data_df['job_id'].values
        self.labels = data_df['status'].values
        print(f'Loaded {len(self.users)} {data_type} {data_split} examples.')

    def get_entity_id_list(self, id, id_type):
        
        # print(f'self.id_to_entity_id_memmap_index: {self.id_to_entity_id_memmap_index}') # DEBUG
        
        if str(id) in self.id_to_entity_id_memmap_index[id_type]:
            return self.id_to_entity_id_memmap_index[id_type][str(id)]
        return self.id_to_entity_id_memmap_index[id_type]['0']
    
    def get_entity_id_memmap(self, id, id_type):
        
        # print(f'id: {id}') # DEBUG
        # print(f'id type: {id_type}') # DEBUG
        # print(f'entity id list: {self.get_entity_id_list(id, id_type)}') # DEBUG
        
        try:
            return self.memmap[self.get_entity_id_list(id, id_type)]
        except IndexError:
            print(f'IndexError: {id} {id_type}')
            print(f'entity id list: {self.get_entity_id_list(id, id_type)}')
            print(f'shape of memmap: {self.memmap.shape}')
            raise IndexError
    
    # def get_entity_id_memmap_batch(self, id_list, id_type):
        
    #     print(f'id_list: {id_list}') # DEBUG
    #     print(f'id_list.shape: {id_list.shape}') # DEBUG
    #     print(f'id type: {id_type}') # DEBUG
        
    #     return torch.tensor([self.get_entity_id_memmap(id, id_type) for id in id_list])

    def __getitem__(self, index):
        return (self.get_entity_id_memmap(self.users[index], 'user'), self.get_entity_id_memmap(self.jobs[index], 'job')), self.labels[index]
    
    def __len__(self):
        return len(self.users)
        
def get_max_entity_length(encoded_data_dir):
    """
    Max entity length is needed to initialise the memmap shape. 
    This info is saved in the config file in the encoded data dir. 
    """
    with open(f'{encoded_data_dir}config.json', 'r') as infile:
        config = json.load(infile)
    return config['max_entity_length']

def get_config(encoded_data_dir):
    """
    Straight up returns the whole config. 
    We need this for max_entity_length and n_entities, 
    required for initialising the memmap shapes. 
    """
    with open(f'{encoded_data_dir}config.json', 'r') as infile:
        config = json.load(infile)
    return config
        
def load_encoded_data_TribePad(
    data_dir: str, 
    encoded_data_dir: str, 
    data_type: str, 
    batch_size: int, 
    num_steps = 50
    ):
    """Return data iterators of the encoded TribePad dataset."""
    
    num_workers = get_dataloader_workers()
    # max_entity_length = get_max_entity_length(encoded_data_dir)
    config = get_config(encoded_data_dir)
    
    data_dict = {
        data_split: EncodedDataset(
            root_data_dir = data_dir, 
            encoded_data_dir = encoded_data_dir, 
            data_type = data_type, 
            data_split = data_split,
            n_ids = config['n_ids'], 
            max_entity_length = config['max_entity_length'],
            ) for data_split in ['train', 'test', 'dev']
    }
    iter_dict = {
        data_split: torch.utils.data.DataLoader(
            data_dict[data_split],
            batch_size, 
            shuffle = True, 
            num_workers = num_workers
        ) for data_split in ['train', 'test', 'dev']
    }
    return iter_dict['train'], iter_dict['dev'], iter_dict['test']