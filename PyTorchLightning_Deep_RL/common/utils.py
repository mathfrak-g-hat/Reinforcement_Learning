#####################
##### UTILITIES #####
#####################
## AJ Zerouali
## Updated 23/06/30

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as th
from torch import nn

'''
    NEURAL NET CONSTRUCTION HELPER FUNCTION
    23/06/30
'''
def make_net_modules(input_dim: int,
                     output_dim: int,
                     net_arch: list,
                     dropout_probs: list,
                     layer_activation_fn: nn.Module,
                     output_activation_fn: nn.Module,
                     weight_init_mthd: str = "",
                     weight_init_seed: int = 123,
                    ):
    '''
        Helper function to build a neural network with a specified
        architecture, allowing for dropout layers with distinct probabilities.
        Outputs the list of torch modules required for the desired neural net, 
        eventually obtained using:
            net = nn.Sequential(*net_modules).
        
        :param input_dim: int. Number of units in the input layer.
        :param input_dim: int. Number of units in the output layer.
        :param net_arch: list of int. Number of units in each hidden layer.
        :param dropout_probs: list of float. List of dropout probabilities for hidden
            layers. If empty, no dropout layers are added. 
            Must have the same length as the net_arch list.
        :param layer_activation_fn: nn.Module. Activation function for hidden layers.
        :param output_activation_fn: nn.Module. Activation function for output layer. 
            If equals nn.Identity or None, no activation function is added after the output layer.
        :param weight_init_mthd: str, weight initialization method. A string from the 
            following list: ["Xavier_normal", "Xavier_uniform", "Kaiming_normal", "Kaiming_uniform"]
        :param weight_init_seed: int = 123, weight initialization seed.
            
        :return net_modules: list of torch objects to be used as input to instantiate 
            a neural net. 
    '''
    # Verifications
    if len(net_arch) >0:
        use_hidden_lyrs = True
        ## Check dropout probability list
        if len(dropout_probs)>0:
            ## Check dropout for hidden layers
            if len(dropout_probs) != len(net_arch):
                raise ValueError("dropout_probs must have the same length as net_arch")
            else:
                use_dropout = True
        else:
            use_dropout = False
    else:
        use_hidden_lyrs = False
        
    ## Output layer
    if output_dim == 0:
        if not use_hidden_lyrs:
            raise ValueError("out_dim cannot be 0 if there are no hidden layers")
        else:
            use_output_layer = False
    else:
        use_output_layer = True
        
    # Weight initialization type
    ### Temporary: Supports Xavier and Kaiming only for now
    admissible_initializations = ["Xavier_normal", "Xavier_uniform", 
                                  "Kaiming_normal", "Kaiming_uniform"]
    no_wt_initialization_list = [None, "", "none", "None"]
    if weight_init_mthd not in no_wt_initialization_list:
        
        # Verify that init. method is supported
        if weight_init_mthd not in admissible_initializations:
            raise NotImplementedError(f"Unsupported weight initialization method.\n"\
                                      f"Supported methods: {admissible_initializations}"
                                     )
        
        # Init. Boolean
        initialize_layer_weights = True
        
        # Set seed
        th.manual_seed(weight_init_seed)
        
        # Weight initialization dict
        activation_alias_dict={nn.ReLU: "relu", nn.Tanh: "tanh",
                               nn.Sigmoid: "sigmoid", nn.SELU: "selu",
                               nn.Identity: "linear", nn.Linear: "linear",
                              }
        
        # Get layer activation and output activation gains
        layer_activation_gain = 0.0
        output_activation_gain = 0.0
        if output_activation_fn == None:
            output_activation_gain = 1.0
        for activation_class in list(activation_alias_dict.keys()):
            # Get layer activation gain:
            if isinstance(layer_activation_fn(), activation_class):
                layer_activation_name = activation_alias_dict[activation_class]
                layer_activation_gain = nn.init.calculate_gain(layer_activation_name)
            # Get output activation gain:
            if isinstance(output_activation_fn(), activation_class):
                output_activation_name = activation_alias_dict[activation_class]
                output_activation_gain = nn.init.calculate_gain(output_activation_name)
                
        # Define weight initialization function
        if weight_init_mthd == "Xavier_normal":
            def init_layer_wts(layer_weights, gain, is_output_layer):
                return nn.init.xavier_normal_(tensor = layer_weights, gain = gain)
        elif weight_init_mthd == "Xavier_uniform":
            def init_layer_wts(layer_weights, gain, is_output_layer):
                return nn.init.xavier_uniform_(tensor = layer_weights, gain = gain)
        if weight_init_mthd == "Kaiming_normal":
            def init_layer_wts(layer_weights, gain, is_output_layer):
                if is_output_layer:
                    return nn.init.kaiming_normal_(tensor = layer_weights, 
                                                    mode= "fan_in", 
                                                    nonlinearity = output_activation_name)
                else:
                    return nn.init.kaiming_normal_(tensor = layer_weights, 
                                                    mode= "fan_in", 
                                                    nonlinearity = layer_activation_name)
        elif weight_init_mthd == "Kaiming_uniform":
            def init_layer_wts(layer_weights, gain, is_output_layer):
                if is_output_layer:
                    return nn.init.kaiming_uniform_(tensor = layer_weights, 
                                                    mode= "fan_in", 
                                                    nonlinearity = output_activation_name)
                else:
                    return nn.init.kaiming_uniform_(tensor = layer_weights, 
                                                    mode= "fan_in", 
                                                    nonlinearity = layer_activation_name)
                    
        
    else:
        # Init. Boolean
        initialize_layer_weights = False

        
    # Make net_modules list 
    net_modules = []
    if not use_hidden_lyrs:
        # Shallow 
        layer = nn.Linear(input_dim, output_dim)
        # Init. layer weights
        if initialize_layer_weights:
            init_layer_wts(layer.weight, gain = output_activation_gain,
                           is_output_layer = True)
        net_modules.append(layer)
        net_modules.append(output_activation_fn())
    else:
        ## First hidden layer + activation + dropout
        layer = nn.Linear(input_dim, net_arch[0])
        # Init. layer weights
        if initialize_layer_weights:
            init_layer_wts(layer.weight, gain = layer_activation_gain,
                           is_output_layer = False)
        net_modules.append(layer)
        net_modules.append(layer_activation_fn())
        if use_dropout:
            net_modules.append(nn.Dropout(dropout_probs[0]))
        ## Remaining hidden layers from net_arch + activation + dropout
        for idx in range(len(net_arch)-1):
            layer = nn.Linear(net_arch[idx], net_arch[idx+1])
            # Init. layer weights
            if initialize_layer_weights:
                init_layer_wts(layer.weight, gain = layer_activation_gain,
                               is_output_layer = False)
                #nn.init.xavier_uniform_(layer.weight, gain = layer_activation_gain)
            net_modules.append(layer)
            net_modules.append(layer_activation_fn())
            if use_dropout:
                net_modules.append(nn.Dropout(dropout_probs[idx+1]))
        ## Output layer + activation
        if use_output_layer:
            layer = nn.Linear(net_arch[-1], output_dim)
            # Init. layer weights
            if initialize_layer_weights:
                init_layer_wts(layer.weight, gain = output_activation_gain,
                               is_output_layer = True)
            net_modules.append(layer)
        if (output_activation_fn != None) and (output_activation_fn != nn.Identity):
            net_modules.append(output_activation_fn())
            
    # Output
    return net_modules