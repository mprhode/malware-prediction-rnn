""" 
Hyperparameter configurations stored as dictionaries
Random search space + 3 configurations of 
RNN reported in paper https://arxiv.org/pdf/1708.03513.pdf
"""
import numpy as np
from .useful import merge_two_dicts

#Parameters fixed for all experiments
fixed_parameters = { 
    "layer_type": ["GRU"],          # recurrent_unit
    "loss": ["binary_crossentropy"], #loss function
    "kernel_initializer": ["lecun_uniform"],  #initialise input weights with recommended from http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    "recurrent_initializer": ["lecun_uniform"], # initialise recurrent weights with recommended from http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    "activation": ["sigmoid"],      # Activation functin of Dense output layer
    "recurrent_dropout": [0]
}

def get_all():
    """Return dict of possible values, 
    possible values can be stored as a list --> equal probability of chosing any option
    or as a dictionary with relative probabilities stored to weight choices e.g. for weight merge_two_dicts rule"""
    all_options =  {
        "depth": [1,2,3],           # number of hidden layers
        "bidirectional": [True, False], # bidirectional hidden layers
        "hidden_neurons": list(range(1, 501)), #number of hidden neurons
        "optimiser": ["adam"],
        "dropout": [0] * 3 + [0.1,0.2,0.3,0.4,0.5], #list(np.arange(0, 0.51, 0.1)), #dropout rate
        "b_l1_reg": [0, 0.01],          # bias regulariser (l1)
        "b_l2_reg": [0, 0.01],          # bias regulariser (l2)
        "r_l1_reg": [0, 0.01],          #  weight regulariser (l1)
        "r_l2_reg": [0, 0.01],          #  weight regulariser (l2)
        "epochs": list(range(1, 100)), #501)),  # number of training epochs
        "sequence_length": list(range(1, 21)), 
        "batch_size": [32*(2**i) for i in range(5)] ,
        "recurrent_dropout": [0] * 3 + [0.1,0.2,0.3,0.4,0.5]
    }
    return merge_two_dicts(all_options, fixed_parameters)
        
def get_A():
    """Return dictionary of parameters for 
    config A in https://arxiv.org/pdf/1708.03513.pdf"""
    A = {
        "depth": [3],           
        "bidirectional": [True], 
        "hidden_neurons": [74], 
        "learning_rate": [0.001], 
        "optimiser": ["adam"],
        "dropout": [0.3],           
        "b_l1_reg": [0],                
        "b_l2_reg": [0],                
        "r_l1_reg": [0],                
        "r_l2_reg": [0.01],                
        "epochs": [53],  
        "sequence_length": list(range(1, 31)), 
        "batch_size": [64], 
        "description": ["A"],
    }
    return merge_two_dicts(A, fixed_parameters)

def get_B():
    """Return dictionary of parameters for 
    config B in https://arxiv.org/pdf/1708.03513.pdf"""
    B = {
        "depth": [1],           
        "bidirectional": [True],
        "hidden_neurons": [358], 
        "learning_rate": [0.001], 
        "optimiser": ["adam"],
        "dropout": [0.1],           
        "b_l1_reg": [0],                
        "b_l2_reg": [0],                
        "r_l1_reg": [0],                
        "r_l2_reg": [0.01],                
        "epochs": [112], 
        "sequence_length": list(range(1, 21)), 
        "batch_size": [64], 
        "description": ["B"],
    }
    return merge_two_dicts(B, fixed_parameters)

def get_C():
    """Return dictionary of parameters for 
    config C in https://arxiv.org/pdf/1708.03513.pdf"""
    C = {
        "depth": [2],           
        "bidirectional": [False],
        "hidden_neurons": [195], 
        "learning_rate": [0.001], 
        "optimiser": ["adam"],
        "dropout": [0.1],       
        "b_l1_reg": [0],                
        "b_l2_reg": [0],                
        "r_l1_reg": [0.01],                
        "r_l2_reg": [0],                
        "epochs": [39],  
        "sequence_length": list(range(1, 31)), 
        "batch_size": [64], 
        "description": ["C"],
    }
    return merge_two_dicts(C, fixed_parameters)

def get_A_B_C():
    """return top 3 configurations as dictionary values"""
    return {
        "A": get_A(),
        "B": get_B(),
        "C": get_C(),
    }