import json
import numpy as np
import torch
import os
import warnings

def load_config(path):
    f = open(path)
    json_data = json.load(f)
    f.close()
    return json_data

def set_seed(seed=123):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed) # Set a fixed value for the hash seed

def calc_grad_norm(model):
    grad_sum_sqrd = 0
    for param in model.parameters():
        if param.grad is not None:
            grad_sum_sqrd += torch.sum((param.grad.detach().clone().flatten())**2)

    norm = torch.sqrt(grad_sum_sqrd)
    return norm

'''for each prediction, calculates the entropy (of the predicted distribution over all tokens), and then calculates the mean over all predictions in the batch'''
def calc_mean_entropy(predicted_logits):
    vocab_size = predicted_logits.shape[2]
    probabilities = torch.softmax(predicted_logits.reshape(-1, vocab_size), axis=1)
    prob_zeros_mask = probabilities == 0.
    tmp = probabilities * torch.log2(probabilities) # when a probability equals 0 this gives 0*-inf and torch returns nan. by the entropy definition it should equal 0, so we fix that
    tmp[prob_zeros_mask] = 0.
    if torch.any(torch.isnan(tmp)):
        warnings.warn("Warning: entropy calculation (metric) has nans in it")

    entropy = -torch.sum(tmp, axis=1)
    return torch.mean(entropy)