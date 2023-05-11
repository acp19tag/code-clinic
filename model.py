import torch
from torch import nn
from torch.nn import functional as F
from dataloader import *

def mlp(num_inputs, num_hiddens, flatten, dropout):  # sourcery skip: merge-list-append
    
    # print('num_inputs: ', num_inputs)   # DEBUG
    # print('num_hiddens: ', num_hiddens) # DEBUG
    # print('flatten: ', flatten)         # DEBUG
    # print('dropout: ', dropout)         # DEBUG
    
    net = []
    net.append(nn.Dropout(dropout)) 
    net.append(nn.Linear(num_inputs, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    net.append(nn.Dropout(dropout)) 
    net.append(nn.Linear(num_hiddens, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    return nn.Sequential(*net)

class Attend(nn.Module):
    
    def __init__(self, num_inputs, num_hiddens, dropout, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_inputs, num_hiddens, flatten=False, dropout = dropout)

    def forward(self, A, B):
        # Shape of `A`/`B`: (`batch_size`, no. of tokens in sequence A/B,
        # `embed_size`)
        # print()
        # print('A shape: ', A.shape) # DEBUG
        # print('A dtype: ', A.dtype) # DEBUG
        # Shape of `f_A`/`f_B`: (`batch_size`, no. of tokens in sequence A/B,
        # `num_hiddens`)
        # print('B shape: ', B.shape) # DEBUG
        # print('B dtype: ', B.dtype) # DEBUG
        # print()
        
        f_A = self.f(A)
        f_B = self.f(B)
        # Shape of `e`: (`batch_size`, no. of tokens in sequence A,
        # no. of tokens in sequence B)
        
        # print('f_A shape: ', f_A.shape) # DEBUG
        # print('f_B shape: ', f_B.shape) # DEBUG
        # print()
        
        e = torch.bmm(f_A, f_B.permute(0, 2, 1))
        # Shape of `beta`: (`batch_size`, no. of tokens in sequence A,
        # `embed_size`), where sequence B is softly aligned with each token
        # (axis 1 of `beta`) in sequence A
        beta = torch.bmm(F.softmax(e, dim=-1), B)
        # Shape of `alpha`: (`batch_size`, no. of tokens in sequence B,
        # `embed_size`), where sequence A is softly aligned with each token
        # (axis 1 of `alpha`) in sequence B
        alpha = torch.bmm(F.softmax(e.permute(0, 2, 1), dim=-1), A)
        return beta, alpha
    
class Compare(nn.Module):
    def __init__(self, num_inputs, num_hiddens, dropout, **kwargs):
        
        # DEBUG ATTRIBUTES
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_inputs, num_hiddens, flatten=False, dropout = dropout)

    def forward(self, A, B, beta, alpha):
        
        # print('Forward pass.') # DEBUG
        # print('A shape: ', A.shape)
        # print('B shape: ', B.shape)
        # print('beta shape: ', beta.shape)
        # print('alpha shape: ', alpha.shape)
        
        # print('num inputs: ', self.num_inputs)
        # print('num hiddens: ', self.num_hiddens)
        
        V_A = self.g(torch.cat([A, beta], dim=2))
        V_B = self.g(torch.cat([B, alpha], dim=2))
        return V_A, V_B
    
class Aggregate(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs, dropout, **kwargs):
        
        # DEBUG ATTRIBUTES
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_outputs = num_outputs
        
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_inputs, num_hiddens, flatten=True, dropout = dropout)
        self.linear = nn.Linear(num_hiddens, num_outputs)

    def forward(self, V_A, V_B):
        # Sum up both sets of comparison vectors
        
        # print('V_A shape: ', V_A.shape) # DEBUG
        # print('V_B shape: ', V_B.shape) # DEBUG
        
        V_A = V_A.sum(dim=1)
        V_B = V_B.sum(dim=1)
        
        # print('V_A shape after sum dim 1: ', V_A.shape) # DEBUG
        # print('V_B shape after sum dim 1: ', V_B.shape) # DEBUG
        
        # print(f'num inputs: {self.num_inputs}')
        # print(f'num hidden: {self.num_hiddens}')
        # print(f'num outputs: {self.num_outputs}')
        
        # Feed the concatenation of both summarization results into an MLP
        return self.linear(self.h(torch.cat([V_A, V_B], dim=1))) # Y_hat
    
class DecomposableAttention(nn.Module):
    def __init__(self, num_hiddens, num_inputs_attend,
                 num_inputs_compare, num_inputs_agg, dropout,
                 root_data_dir, # for EntityLoader
                 **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        # self.embedding = nn.Embedding(len(vocab), embed_size)
        self.embedding = EntityLoader(root_data_dir)
        self.attend = Attend(num_inputs_attend, num_hiddens, dropout = dropout)
        self.compare = Compare(num_inputs_compare, num_hiddens, dropout = dropout)
        # There are 2 possible outputs: hired/interviewed (1) or not (0)
        self.aggregate = Aggregate(num_inputs_agg, num_hiddens, num_outputs=2, dropout = dropout)

    def forward(self, X):
        
        user_ids, job_ids = X
        A = self.embedding.get_entities(user_ids)
        B = self.embedding.get_entities(job_ids)
        # A, B = X
        
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        return self.aggregate(V_A, V_B) # Y_hat